from enum import IntEnum

import torch

from refiner.utils.lie import SE3_exp, SE3_log, inverse_transform
from refiner.layers.lie import Se3_left_action, So3_left_action, Se3_right_action, So3_right_action
class AxisAngleBackend():
    def __init__(self,
        mat: torch.Tensor,
        requires_grad: bool = False,
    ):
        self.params = torch.zeros(6, dtype=mat.dtype, device=mat.device, requires_grad = requires_grad)
        self._base_matrix = mat.detach()

    def update_params(self):
        if torch.isnan(self.params).sum() > 0:
            return 1
        self._base_matrix = self.convert_to_mat().detach()
        self.params.data.fill_(0)
        return 0
    
    @property
    def tau(self) -> torch.Tensor:
        return self.params

    @property
    def theta(self) -> torch.Tensor:
        return self.params[3:]

    @property
    def rho(self) -> torch.Tensor:
        return self.params[:3]
    
    @classmethod
    def convert_from_mat(cls, mat: torch.Tensor) -> torch.Tensor:
        return SE3_log(mat.unsqueeze(0)).squeeze(0)
    
    def convert_to_mat(self) -> torch.Tensor:
        exp = SE3_exp(self.tau.unsqueeze(0)).squeeze(0)
        return exp @ self._base_matrix # = (T^-1 @ Exp(-tau))^-1

    def apply_transform(self, points: torch.Tensor) -> torch.Tensor:
        mat = self.convert_to_mat().detach().unsqueeze(0)
        points = Se3_left_action.apply(mat, points, self.tau.unsqueeze(0))
        points = points[:, 0]
        return points
    
    def apply_inverse_transform(self, points: torch.Tensor) -> torch.Tensor:
        mat = self.convert_to_mat().unsqueeze(0)
        mat = inverse_transform(mat).detach()
        points = Se3_right_action.apply(mat, points, -self.tau.unsqueeze(0))
        points = points[:, 0]
        return points

    def apply_rotation(self, points: torch.Tensor) -> torch.Tensor:
        mat = self.convert_to_mat().detach()[:3, :3].unsqueeze(0)
        points = So3_left_action.apply(mat, points, self.theta.unsqueeze(0))
        points = points[:, 0]
        return points
        
    def apply_inverse_rotation(self, points: torch.Tensor) -> torch.Tensor:
        mat = self.convert_to_mat()[:3, :3]
        mat = mat.T.unsqueeze(0).detach()
        points = So3_right_action.apply(mat, points, -self.theta.unsqueeze(0))
        points = points[:, 0]
        return points


    
class IntrinsicsBackend():
    class MODEL(IntEnum):
        F = 1
        FXFY = 2
        FXFYCXCY = 4

    class IntrParam(IntEnum):
        fx = 0
        fy = 1
        cx = 2
        cy = 3

    optim_dict = {
        MODEL.F: {
            IntrParam.fx: 0,
            IntrParam.fy: 0
        },
        MODEL.FXFY: {
            IntrParam.fx: 0,
            IntrParam.fy: 1
        },
        MODEL.FXFYCXCY: {
            IntrParam.fx: 0,
            IntrParam.fy: 1,
            IntrParam.cx: 2,
            IntrParam.cy: 3
        }
    }

    frozen_dict = {
        MODEL.F: {
            IntrParam.cx: 0,
            IntrParam.cy: 1
        },
        MODEL.FXFY: {
            IntrParam.cx: 0,
            IntrParam.cy: 1
        },
        MODEL.FXFYCXCY: {
        }
    }
    def __init__(self, 
                 mat: torch.Tensor,
                 model: MODEL,
                 factor: float = 1.,
                 requires_grad: bool = False):

        self._model = model
        self.factor = torch.tensor(factor, device = mat.device, dtype = mat.dtype)
        mat = mat.detach().clone()
        cx = (mat[0, 2] / self.factor).detach().clone()
        cy = (mat[1, 2] / self.factor).detach().clone()

        self.params = []
        self.frozen_params = []

        match self._model:
            case IntrinsicsBackend.MODEL.F:
                f = ((mat[0, 0] + mat[1, 1]) / 2 / self.factor).detach().clone()
                self.params.append(f)
                self.frozen_params += [cx, cy]
            case _:
                fx = (mat[0, 0] / self.factor).detach().clone()
                fy = (mat[1, 1] / self.factor).detach().clone()
                self.params += [fx, fy]
                match self._model:
                    case IntrinsicsBackend.MODEL.FXFY:
                        self.frozen_params += [cx, cy]
                    case _: 
                        self.params += [cx, cy]
        
        self.params = torch.stack(self.params).detach().requires_grad_(requires_grad)
        self._prev_params = torch.zeros_like(self.params)
        if len(self.frozen_params) > 0:
            self.frozen_params = torch.stack(self.frozen_params).detach()
        else:
            self.frozen_params = torch.zeros(0, dtype = mat.dtype, device = mat.device)

    def fetch_param(self, param: IntrParam) -> torch.Tensor:
        for p, d in zip([self.params, self.frozen_params], [self.optim_dict, self.frozen_dict]):
            if param in d[self._model]:
                return p[d[self._model][param]]
    
    def update_params(self):
        self._prev_params = self.params.detach().clone()
    
    @property
    def params_step(self) -> torch.Tensor:
        return self.params - self._prev_params

    @property
    def fx(self) -> torch.Tensor:
        return self.fetch_param(IntrinsicsBackend.IntrParam.fx) * self.factor
    
    @property
    def fy(self) -> torch.Tensor:
        return self.fetch_param(IntrinsicsBackend.IntrParam.fy) * self.factor

    
    @property
    def cx(self) -> torch.Tensor:
        return self.fetch_param(IntrinsicsBackend.IntrParam.cx) * self.factor
    
    @property
    def cy(self) -> torch.Tensor:
        return self.fetch_param(IntrinsicsBackend.IntrParam.cy) * self.factor


class DistortionBackend():
    class MODEL(IntEnum):
        DIV = 0

    def __init__(self,
                 params: torch.Tensor,
                 model: MODEL,
                 require_grad: bool = False):

        self._model = model
        self.params = params.detach().requires_grad_(require_grad)
        self._prev_params = torch.zeros_like(params)

    def update_params(self):
        self._prev_params = self.params.detach().clone()
    
    @property
    def params_step(self) -> torch.Tensor:
        return self.params - self._prev_params