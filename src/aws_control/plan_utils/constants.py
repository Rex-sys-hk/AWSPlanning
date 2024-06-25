import math

class VehicleConst:
    no_wheel_constrain_mode = False
    no_smooth_mode = False
    follow_with_constrain = True
    T = 20
    dt = .3

    WHEEL_BASE = 2.520
    WHEEL_WHIDTH = 1.100
    WHEEL_RADIUS = 0.25
    
    HEIGHT = 1.5

    MASS_Center_offset_x = 0.
    MASS_Center_offset_y = 0.

    MAX_V = 2.5
    MAX_dyaw = math.pi/2
    
    MAX_ACC = 2.5
    MAX_dACC = 2.5
    MAX_ddyaw = math.pi/4

    MAX_STEER_DEG = 75
    MAX_STEER = math.pi/180*MAX_STEER_DEG
    MAX_dSTEER = math.pi/2

    MAX_WHEEL_V = MAX_V/WHEEL_RADIUS
    MAX_WHEEL_ACC = MAX_ACC/WHEEL_RADIUS
    
    half_l = WHEEL_BASE/2
    half_w = WHEEL_WHIDTH/2
    wheel_frames = ['FLaxis','FRaxis','RLaxis','RRaxis']
    _control_center_offset = -half_l
    # _control_center_offset = 0.
    wheel_positions = [[half_l, half_w],
                       [half_l, -half_w],
                       [_control_center_offset, half_w],
                       [_control_center_offset, -half_w]]
    # anticlockwise feasible steer limits
    _front = MAX_STEER
    _rear = MAX_STEER
    # _rear = 1.e-3
    wheel_steer_limits = [[-_front,_front],
                          [-_front,_front],
                          [-_rear,_rear],
                          [-_rear,_rear]]
    wheel_num = len(wheel_positions)

    footprint_length = 3.5
    footprint_width = 1.9

    cellSize = 1
    headings = 8
    deltaHeadingRad = 2*math.pi/headings


    ## collision circle
    OCC_RANGE = 30
    ## collision circle radius
    ccr = math.sqrt(WHEEL_WHIDTH**2/2)
    ## collision circle center list
    cccl = [WHEEL_BASE/2-WHEEL_WHIDTH/2,0.,-WHEEL_BASE/2+WHEEL_WHIDTH/2]
    occ_r = math.sqrt(2*cellSize**2)/2



