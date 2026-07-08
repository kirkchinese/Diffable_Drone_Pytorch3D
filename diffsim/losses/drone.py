"""Legacy DroneLoss construction kept behind the common loss factory."""

from loss import DroneLoss

from .factory import LossBuildContext, register_loss


def build_drone_navigation_loss(context: LossBuildContext) -> DroneLoss:
    args = context.args
    return DroneLoss(
        coef_v=args.coef_v,
        coef_speed=args.coef_speed,
        coef_v_pred=args.coef_v_pred,
        coef_collide=args.coef_collide,
        coef_obj_avoidance=args.coef_obj_avoidance,
        coef_d_acc=args.coef_d_acc,
        coef_d_jerk=args.coef_d_jerk,
        coef_d_snap=args.coef_d_snap,
        coef_ground_affinity=args.coef_ground_affinity,
        coef_bias=args.coef_bias,
        coef_lateral=getattr(args, "coef_lateral", 0.0),
        coef_drone_collide=getattr(args, "coef_drone_collide", 5.0),
        ctl_dt=context.control_dt,
        window_size=getattr(args, "window_size", 30),
        loss_v_mode=getattr(args, "loss_v_mode", "mse"),
        adaptive_decay_rate=getattr(args, "adaptive_decay_rate", 2.0),
        ga_z_ceiling=getattr(args, "ga_z_ceiling", 5.0),
    )


register_loss("drone_navigation", build_drone_navigation_loss)
