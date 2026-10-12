import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from .homography_utils import warp_points, warp_points_torch
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
import pprint


# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def warp_pts_error(H_est, pts_coord, H_gt=None, pts_gt=None):
    """
    corner_coord: 4*2
    """
    if H_gt is not None:
        est_warp = warp_points(pts_coord, H_est, False)
        est_gt = warp_points(pts_coord, H_gt, False)
        diff = est_warp - est_gt
    elif pts_gt is not None:
        est_warp = warp_points(pts_coord, H_est, False)
        diff = est_warp - pts_gt

    return np.mean(np.linalg.norm(diff, axis=1))

def homo_warp_match_distance(H_gt, kpts0, kpts1, hw):
    """
    corner_coord: 4*2
    """
    if isinstance(H_gt, np.ndarray):
        kpts_warped = warp_points(kpts0, H_gt)
        normalized_distance = np.linalg.norm((kpts_warped - kpts1) / hw[None, [1,0]], axis=1)
    else:
        kpts_warped = warp_points_torch(kpts0, H_gt)
        normalized_distance = torch.linalg.norm((kpts_warped - kpts1) / hw[None, [1,0]], axis=1)
    return normalized_distance

def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data, config):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f'].clone().detach()

    if config.LOFTR.FINE.MTD_SPVS:
        m_bids = data['m_bids_f'] if 'm_bids_f' in data else data['m_bids']
    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})

def compute_homo_match_warp_errors(data, config):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    
    homography_gt = data['homography']
    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']
    origin_img0_size = data['origin_img_size0']

    if config.LOFTR.FINE.MTD_SPVS:
        m_bids = data['m_bids_f'] if 'm_bids_f' in data else data['m_bids']
    epi_errs = []
    for bs in range(homography_gt.shape[0]):
        mask = m_bids == bs
        epi_errs.append(
            homo_warp_match_distance(homography_gt[bs], pts0[mask], pts1[mask], origin_img0_size[bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})

    
def compute_symmetrical_epipolar_errors_gt(data, config):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f_gt']
    pts1 = data['mkpts1_f_gt']

    epi_errs = []
    for bs in range(Tx.size(0)):
        # mask = m_bids == bs
        assert bs == 0
        mask = torch.tensor([True]*pts0.shape[0], device = pts0.device)
        if config.LOFTR.FINE.MTD_SPVS:
            epi_errs.append(
                symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
        else:
            epi_errs.append(
                symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret

def estimate_homo(kpts0, kpts1, thresh, conf=0.99999, mode='affine'):
    if mode == 'affine':
        H_est, inliers = cv2.estimateAffine2D(kpts0, kpts1, ransacReprojThreshold=thresh, confidence=conf, method=cv2.RANSAC)
        if H_est is None:
            return np.eye(3) * 0, np.empty((0))
        H_est = np.concatenate([H_est, np.array([[0, 0, 1]])], axis=0) # 3 * 3
    elif mode == 'homo':
        H_est, inliers = cv2.findHomography(kpts0, kpts1, method=cv2.LMEDS, ransacReprojThreshold=thresh)
        if H_est is None:
            return np.eye(3) * 0, np.empty((0))

    return H_est, inliers

def compute_homo_corner_warp_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N] # Actually warp error
            "t_errs" List[float]: [N] # Zero, place holder
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})
    
    if config.LOFTR.FINE.MTD_SPVS:
        m_bids = data['m_bids_f'].cpu().numpy() if 'm_bids_f' in data else data['m_bids'].cpu().numpy()

    else:
        m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    homography_gt = data['homography'].cpu().numpy()
    origin_size_0 = data['origin_img_size0'].cpu().numpy()

    for bs in range(homography_gt.shape[0]):
        mask = m_bids == bs
        ret = estimate_homo(pts0[mask], pts1[mask], pixel_thr, conf=conf)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            H_est, inliers = ret
            corner_coord = np.array([[0, 0], [0, origin_size_0[bs][0]], [origin_size_0[bs][1], 0], [origin_size_0[bs][1], origin_size_0[bs][0]]])
            corner_warp_distance = warp_pts_error(H_est, corner_coord, H_gt=homography_gt[bs])
            data['R_errs'].append(corner_warp_distance)
            data['t_errs'].append(0)
            data['inliers'].append(inliers)

def compute_warp_control_pts_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N] # Actually warp error
            "t_errs" List[float]: [N] # Zero, place holder
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})
    
    if config.LOFTR.FINE.MTD_SPVS:
        m_bids = data['m_bids_f'].cpu().numpy() if 'm_bids_f' in data else data['m_bids'].cpu().numpy()

    else:
        m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    gt_2D_matches = data["gt_2D_matches"].cpu().numpy()

    data.update({'epi_errs': torch.zeros(m_bids.shape[0])})
    for bs in range(gt_2D_matches.shape[0]):
        mask = m_bids == bs
        ret = estimate_homo(pts0[mask], pts1[mask], pixel_thr, conf=conf, mode=config.TRAINER.WARP_ESTIMATOR_MODEL)

        if ret is None:
            data['R_errs'].append(np.inf)
            data['t_errs'].append(np.inf)
            data['inliers'].append(np.array([]).astype(bool))
        else:
            H_est, inliers = ret
            img0_pts, img1_pts = gt_2D_matches[bs][:, :2], gt_2D_matches[bs][:, 2:]
            pts_warp_distance = warp_pts_error(H_est, img0_pts, pts_gt=img1_pts)
            print(pts_warp_distance)
            data['R_errs'].append(pts_warp_distance)
            data['t_errs'].append(0)
            data['inliers'].append(inliers)

def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})
    
    if config.LOFTR.FINE.MTD_SPVS:
        m_bids = data['m_bids_f'].cpu().numpy() if 'm_bids_f' in data else data['m_bids'].cpu().numpy()

    else:
        m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        if config.LOFTR.EVAL_TIMES >= 1:
            bpts0, bpts1 = pts0[mask], pts1[mask]
            R_list, T_list, inliers_list = [], [], []
            for _ in range(5):
                shuffling = np.random.permutation(np.arange(len(bpts0)))
                if _ >= config.LOFTR.EVAL_TIMES:
                    continue
                bpts0 = bpts0[shuffling]
                bpts1 = bpts1[shuffling]
                
                ret = estimate_pose(bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf)
                if ret is None:
                    R_list.append(np.inf)
                    T_list.append(np.inf)
                    inliers_list.append(np.array([]).astype(bool))
                    print('Pose error: inf')
                else:
                    R, t, inliers = ret
                    t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
                    R_list.append(R_err)
                    T_list.append(t_err)
                    inliers_list.append(inliers)
                    print(f'Pose error: {max(R_err, t_err)}')
            R_err_mean = np.array(R_list).mean()
            T_err_mean = np.array(T_list).mean()
            # inliers_mean = np.array(inliers_list).mean()

            data['R_errs'].append(R_list)
            data['t_errs'].append(T_list)
            data['inliers'].append(inliers_list[0])

        else:
            ret = estimate_pose(pts0[mask], pts1[mask], K0[bs], K1[bs], pixel_thr, conf=conf)

            if ret is None:
                data['R_errs'].append(np.inf)
                data['t_errs'].append(np.inf)
                data['inliers'].append(np.array([]).astype(bool))
                print('Pose error: inf')
            else:
                R, t, inliers = ret
                t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
                data['R_errs'].append(R_err)
                data['t_errs'].append(t_err)
                data['inliers'].append(inliers)
                print(f'Pose error: {max(R_err, t_err)}')


# --- METRIC AGGREGATION ---
def error_rmse(error):
    squard_errors = np.square(error) # N * 2
    mse = np.mean(np.sum(squard_errors, axis=1))
    rmse = np.sqrt(mse)
    return rmse

def error_mae(error):
    abs_diff = np.abs(error) # N * 2
    absolute_errors = np.sum(abs_diff, axis=1)
    
    # Return the maximum absolute error
    mae = np.max(absolute_errors)
    return mae

def error_auc(errors, thresholds, method='exact_auc'):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    if method == 'exact_auc':
        errors = [0] + sorted(list(errors))
        recall = list(np.linspace(0, 1, len(errors)))

        aucs = []
        for thr in thresholds:
            last_index = np.searchsorted(errors, thr)
            y = recall[:last_index] + [recall[last_index-1]]
            x = errors[:last_index] + [thr]
            aucs.append(np.trapz(y, x) / thr)
        return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}
    elif method == 'fire_paper':
        aucs = []
        for threshold in thresholds:
            accum_error = 0
            percent_error_below = np.zeros(threshold + 1)
            for i in range(1, threshold + 1):
                percent_error_below[i] = np.sum(errors < i) * 100 / len(errors)
                accum_error += percent_error_below[i]
            
            aucs.append(accum_error / (threshold * 100))

        return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}
    elif method == 'success_rate':
        aucs = []
        for threshold in thresholds:
            aucs.append((errors < threshold).astype(float).mean())
        return {f'SR@{t}': auc for t, auc in zip(thresholds, aucs)}
    else:
        raise NotImplementedError


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4, eval_n_time=1, threshold=[5, 10, 20], method='exact_auc'):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = threshold
    if eval_n_time >= 1:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0).reshape(-1, eval_n_time)[unq_ids].reshape(-1)
    else:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    logger.info('num of pose_errors: {}'.format(pose_errors.shape))
    aucs = error_auc(pose_errors, angular_thresholds, method=method)  # (auc@5, auc@10, auc@20)

    if eval_n_time >= 1:
        for i in range(eval_n_time):
            aucs_i = error_auc(pose_errors.reshape(-1, eval_n_time)[:,i], angular_thresholds, method=method)
            logger.info('\n' + f'results of {i}-th RANSAC' + pprint.pformat(aucs_i))
    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)
    
    u_num_mathces = np.array(metrics['num_matches'], dtype=object)[unq_ids]
    u_percent_inliers = np.array(metrics['percent_inliers'], dtype=object)[unq_ids]
    num_matches = {f'num_matches': u_num_mathces.mean() }
    percent_inliers = {f'percent_inliers': u_percent_inliers.mean()}
    return {**aucs, **precs, **num_matches, **percent_inliers}
