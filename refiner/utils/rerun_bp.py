from typing import List
import rerun.blueprint as rrb


def get_blueprint(
    channels: List[str], target_channels: List[str], names: List[str], infer: bool
) -> rrb.Blueprint:
    bp = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    contents=[
                        "+ $origin/points/**",
                        "+ $origin/camera/**",
                        "+ $origin/traj/**",
                    ],
                    origin="/",
                    name="Point Cloud",
                ),
                rrb.Tabs(
                    *[
                        rrb.Spatial2DView(
                            name=f"target_{c}",
                            origin=f"/camera/camera_target/views/{c}",
                            contents=["+ $origin/**"],
                        )
                        for c in target_channels
                    ]
                    + [
                        rrb.Spatial2DView(
                            name=f"final", origin=f"/final", contents=["+ $origin/**"]
                        ),
                        rrb.Spatial2DView(
                            name=f"init", origin=f"/init", contents=["+ $origin/**"]
                        ),
                        rrb.Spatial2DView(
                            name=f"fused",
                            origin=f"/iter/fused",
                            contents=["+ $origin/**"],
                        ),
                        rrb.Spatial2DView(
                            name=f"matchs",
                            origin=f"/iter/matchs",
                            contents=["+ $origin/**"],
                        )
                    ]
                ),
                column_shares=[3, 1],
            ),
            rrb.Horizontal(
                rrb.Tabs(
                    contents=[                      
                        rrb.Spatial2DView(
                            name=f"dist_mask",
                            origin=f"/iter/dist_mask",
                            contents=["+ $origin/**"],
                        )
                    ] + 
                    [
                        rrb.Spatial2DView(
                            origin=f"/iter/{c}", name=c, contents=["+ $origin/**"]
                        )
                        for c in channels
                    ]
                    + [
                        rrb.Spatial2DView(
                            name=f"target_{c}",
                            origin=f"/camera/camera_target/views/{c}",
                            contents=["+ $origin/**"],
                        )
                        for c in target_channels
                    ]
                    + [
                        rrb.Spatial2DView(
                            name=f"target_synth_{c}",
                            origin=f"/camera/camera_target_synth/views/{c}",
                            contents=["+ $origin/**"],
                        )
                        for c in channels
                    ]
                ),
                rrb.Tabs(
                    contents=[
                        rrb.TimeSeriesView(
                            name="loss",
                            origin="/iter/metrics",
                            contents=["+ $origin/loss/**"],
                        ),
                        rrb.TimeSeriesView(
                            name="losses",
                            origin="/iter/metrics",
                            contents=[f"+ $origin/{n}/**" for n in names],
                        ),
                        rrb.TimeSeriesView(
                            name="extrinsics step",
                            origin="/iter/metrics",
                            contents=["+ $origin/extr_step/**"],
                        ),
                        rrb.TimeSeriesView(
                            name="intrinsics step",
                            origin="/iter/metrics",
                            contents=["+ $origin/intr_step/**"],
                        ),
                        rrb.TimeSeriesView(
                                name="distortion",
                                origin="/iter/metrics",
                                contents=["+ $origin/distortion/**"],
                            ),
                    ]
                    + (
                        [
                            rrb.TimeSeriesView(
                                name="rotation",
                                origin="/iter/metrics",
                                contents=["+ $origin/rot_err/**"],
                            ),
                            rrb.TimeSeriesView(
                                name="pixel",
                                origin="/iter/metrics",
                                contents=["+ $origin/pix_err/**"],
                            ),
                            rrb.TimeSeriesView(
                                name="translation",
                                origin="/iter/metrics",
                                contents=["+ $origin/transl_err/**"],
                            ),
                            rrb.TimeSeriesView(
                                name="translation_rescale",
                                origin="/iter/metrics",
                                contents=["+ $origin/transl_rescaled_err/**"],
                            ),
                            rrb.TimeSeriesView(
                                name="focal",
                                origin="/iter/metrics",
                                contents=["+ $origin/focal_err/**"],
                            ),                            
                            rrb.TimeSeriesView(
                                name="center_point",
                                origin="/iter/metrics",
                                contents=["+ $origin/center_point_err/**"],
                            ),
                        ]
                        if not infer
                        else []
                    )
                ),
            ),
        )
    )
    return bp
