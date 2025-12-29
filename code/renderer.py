import cv2
import numpy as np
from config import (W, H, SCALE, FACE_SCALE_MULT, OFFSET_X, OFFSET_Y, 
                    CAMERA_Y_SHIFT, NECK_OFFSET, STATIC_BODY, HAND_CONNECTIONS, FACEMESH_CONNS)

def P(pt): return (int(pt[0]*SCALE)+OFFSET_X, int((pt[1]+CAMERA_Y_SHIFT)*SCALE)+OFFSET_Y)

def PF(pt): 
    f_sc = SCALE * FACE_SCALE_MULT
    return (int(pt[0]*f_sc)+OFFSET_X, int((pt[1]+CAMERA_Y_SHIFT)*f_sc)+OFFSET_Y)

def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))

def draw_frame(fb, ff, ref_b):
    img = np.zeros((H, W, 3), np.uint8)
    
    # Static Body
    for a,b in [(11,12),(11,23),(12,24),(23,24)]: 
        cv2.line(img, P(STATIC_BODY[a]), P(STATIC_BODY[b]), (255,255,255), 3)
    
    # Skeleton Distances
    L_UP_L, L_LO_L = dist((ref_b[44],ref_b[45]),(ref_b[52],ref_b[53])), dist((ref_b[52],ref_b[53]),(ref_b[60],ref_b[61]))
    L_UP_R, L_LO_R = dist((ref_b[48],ref_b[49]),(ref_b[56],ref_b[57])), dist((ref_b[56],ref_b[57]),(ref_b[64],ref_b[65]))

    # Arm Logic
    dir_le = (np.array([fb[52]-fb[44], fb[53]-fb[45]])) / (np.linalg.norm([fb[52]-fb[44], fb[53]-fb[45]])+1e-6)
    dir_re = (np.array([fb[56]-fb[48], fb[57]-fb[49]])) / (np.linalg.norm([fb[56]-fb[48], fb[57]-fb[49]])+1e-6)
    j13 = (STATIC_BODY[12][0]+dir_le[0]*L_UP_L, STATIC_BODY[12][1]+dir_le[1]*L_UP_L)
    j14 = (STATIC_BODY[11][0]+dir_re[0]*L_UP_R, STATIC_BODY[11][1]+dir_re[1]*L_UP_R)
    
    dir_lw = (np.array([fb[60]-fb[52], fb[61]-fb[53]])) / (np.linalg.norm([fb[60]-fb[52], fb[61]-fb[53]])+1e-6)
    dir_rw = (np.array([fb[64]-fb[56], fb[65]-fb[57]])) / (np.linalg.norm([fb[64]-fb[56], fb[65]-fb[57]])+1e-6)
    j15 = (j13[0]+dir_lw[0]*L_LO_L, j13[1]+dir_lw[1]*L_LO_L)
    j16 = (j14[0]+dir_rw[0]*L_LO_R, j14[1]+dir_rw[1]*L_LO_R)

    for s,e in [(P(STATIC_BODY[12]),P(j13)),(P(j13),P(j15)),(P(STATIC_BODY[11]),P(j14)),(P(j14),P(j16))]: 
        cv2.line(img, s, e, (255,255,255), 2)

    # Hands
    for side, w_idx, s_idx in [('l',j15,132),('r',j16,195)]:
        wr_p, root_p = P(w_idx), P((fb[s_idx], fb[s_idx+1]))
        pts = {k: (wr_p[0] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[0] - root_p[0], 
                   wr_p[1] + P((fb[s_idx+k*3], fb[s_idx+k*3+1]))[1] - root_p[1]) for k in range(21)}
        for c in HAND_CONNECTIONS: 
            cv2.line(img, pts[c[0]], pts[c[1]], (0,255,0) if side=='l' else (0,0,255), 2)

    # Face Offset Logic
    mid = ((STATIC_BODY[11][0]+STATIC_BODY[12][0])/2, (STATIC_BODY[11][1]+STATIC_BODY[12][1])/2)
    t_n, s_n = P((mid[0], mid[1]-NECK_OFFSET)), PF((ff[1,0], ff[1,1]))
    dx, dy = t_n[0]-s_n[0], t_n[1]-s_n[1]

    def d_f(c_set, col, th):
        for c in c_set:
            p1, p2 = PF(ff[c[0]]), PF(ff[c[1]])
            cv2.line(img, (p1[0]+dx, p1[1]+dy), (p2[0]+dx, p2[1]+dy), col, th, cv2.LINE_AA)

    d_f(FACEMESH_CONNS.FACEMESH_TESSELATION, (60,60,60), 1)
    d_f(FACEMESH_CONNS.FACEMESH_FACE_OVAL, (255,255,255), 2)
    d_f(FACEMESH_CONNS.FACEMESH_LIPS, (255,255,255), 2)
    d_f(FACEMESH_CONNS.FACEMESH_RIGHT_EYE, (0,0,255), 2)
    d_f(FACEMESH_CONNS.FACEMESH_LEFT_EYE, (0,255,0), 2)
    try: d_f(FACEMESH_CONNS.FACEMESH_RIGHT_IRIS, (0,255,255), 2); d_f(FACEMESH_CONNS.FACEMESH_LEFT_IRIS, (255,255,0), 2)
    except: d_f(FACEMESH_CONNS.FACEMESH_IRISES, (0,255,255), 2)
    
    for idx, col in [(468, (0,255,255)), (473, (255,255,0))]:
        p = PF(ff[idx]); cv2.circle(img, (p[0]+dx, p[1]+dy), 3, col, -1)
    
    return img