import cv2
import math
import numpy as np


# ===============================
# CONFIG
# ===============================
CONF_THRES   = 0.5
SIDE_MARGIN_RATIO = 0.05
TOP_BOTTOM_MARGIN_RATIO = 0.25

MINI_W, MINI_H = 300, 600
DOUBLES_W, COURT_H = 10.97, 23.77   # doubles width, full length
SINGLES_W = 8.23
ALLEY = (DOUBLES_W - SINGLES_W) / 2  # 1.37 m each side

#homography
def matmul(A, B):
    rA, cA = len(A), len(A[0])
    rB, cB = len(B), len(B[0])

    C = [[0]*cB for _ in range(rA)]

    for i in range(rA):
        for j in range(cB):
            for k in range(cA):
                C[i][j] += A[i][k]*B[k][j]

    return C


def transpose(A):
    return list(map(list, zip(*A)))


def normalize_vec(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v]

#gaussian solve
def gaussian_solve(A, b):

    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for i in range(n):

        pivot = M[i][i]

        if abs(pivot) < 1e-9:
            return None
        for j in range(i+1,n):
            if abs(M[j][i]) > abs(pivot):
                M[i],M[j] = M[j],M[i]
                pivot = M[i][i]
                break

        for j in range(i, n+1):
            M[i][j] /= pivot

        for k in range(i+1,n):
            factor = M[k][i]

            for j in range(i,n+1):
                M[k][j] -= factor*M[i][j]

    x=[0]*n

    for i in reversed(range(n)):
        x[i] = M[i][n] - sum(M[i][j]*x[j] for j in range(i+1,n))

    return x

def inverse_iteration(M, iters=60):

    v = normalize_vec([1]*len(M))

    for _ in range(iters):
        sol = gaussian_solve(M, v)
        if sol is None:
            return None
        v = normalize_vec(sol)
    return v

#hartley normalization
def normalize_points(pts):

    pts = np.array(pts)

    cx = np.mean(pts[:,0])
    cy = np.mean(pts[:,1])

    d = np.mean(np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2))

    s = math.sqrt(2)/d

    T = [
        [s,0,-s*cx],
        [0,s,-s*cy],
        [0,0,1]
    ]

    norm=[]

    for x,y in pts:
        norm.append([(x-cx)*s,(y-cy)*s])

    return norm,T

#dlt matrix
def build_A(src,dst):

    A=[]

    for (x,y),(X,Y) in zip(src,dst):

        A.append([-x,-y,-1,0,0,0,x*X,y*X,X])
        A.append([0,0,0,-x,-y,-1,x*Y,y*Y,Y])

    return A

def inv3(M):

    a,b,c=M[0]
    d,e,f=M[1]
    g,h,i=M[2]

    det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)

    return [
        [(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det],
        [(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det],
        [(d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det]
    ]

#normalized dlt  homography
def compute_homography(src_pts,dst_pts):

    src_norm,T1 = normalize_points(src_pts)
    dst_norm,T2 = normalize_points(dst_pts)

    A = build_A(src_norm,dst_norm)

    At = transpose(A)

    AtA = matmul(At,A)

    h = inverse_iteration(AtA)
    if h is None:         
        return None
    
    H = [
        h[0:3],
        h[3:6],
        h[6:9]
    ]

    H = matmul(inv3(T2), matmul(H,T1))

    return np.array(H)

#project points
def project(H,p):

    x,y = p

    px = H[0][0]*x + H[0][1]*y + H[0][2]
    py = H[1][0]*x + H[1][1]*y + H[1][2]
    pz = H[2][0]*x + H[2][1]*y + H[2][2]
    # Guard against zero or near-zero pz
    if abs(pz) < 1e-9:
        return [float('inf'), float('inf')]
    return [px/pz,py/pz]

#ransac
def ransac_homography(src,dst,iters=100,thresh=8):

    src=list(src)
    dst=list(dst)

    bestH=None
    best_inliers=[]

    n=len(src)

    for _ in range(iters):

        idx=np.random.choice(n,4,replace=False)

        s=[src[i] for i in idx]
        d=[dst[i] for i in idx]

        H=compute_homography(s,d)
        if H is None:
             continue
        
        inliers=[]

        for i in range(n):

            p=project(H,src[i])
            #Skip if projection failed
            if math.isinf(p[0]) or math.isinf(p[1]):
                continue

            err=math.sqrt((p[0]-dst[i][0])**2+(p[1]-dst[i][1])**2)

            if err<thresh:
                inliers.append(i)

        if len(inliers)>len(best_inliers):

            best_inliers=inliers
            bestH=H

    if len(best_inliers)>=4:

        s=[src[i] for i in best_inliers]
        d=[dst[i] for i in best_inliers]
        refitted = compute_homography(s, d)
        if refitted is not None:
            bestH = refitted
    return bestH

padding=60

_raw = np.array([
    [0,           0],          # 0  → top-left DOUBLES baseline corner
    [DOUBLES_W,   0],          # 1  → top-right DOUBLES baseline corner
    [0,           COURT_H],    # 2  → bottom-left DOUBLES baseline corner
    [DOUBLES_W,   COURT_H],    # 3  → bottom-right DOUBLES baseline corner
    [ALLEY,       0],          # 4  → top-left singles sideline meets top baseline
    [ALLEY,       COURT_H],    # 5  → bottom-left singles sideline meets bottom baseline
    [ALLEY+SINGLES_W, 0],      # 6  → top-right singles sideline meets top baseline
    [ALLEY+SINGLES_W, COURT_H],# 7  → bottom-right singles sideline meets bottom baseline
    [ALLEY,            6.40],  # 8  → top-left service box corner
    [ALLEY+SINGLES_W,  6.40],  # 9  → top-right service box corner
    [ALLEY,            COURT_H-6.40], # 10 → bottom-left service box corner
    [ALLEY+SINGLES_W,  COURT_H-6.40],# 11 → bottom-right service box corner
    [ALLEY+SINGLES_W/2, 6.40],        # 12 → top T-point
    [ALLEY+SINGLES_W/2, COURT_H-6.40],# 13 → bottom T-point
], dtype=np.float32)

dst_pts = _raw.copy()
dst_pts[:,0] = _raw[:,0] * (MINI_W - 2*padding) / DOUBLES_W + padding
dst_pts[:,1] = _raw[:,1] * (MINI_H - 2*padding) / COURT_H  + padding

###build mini court
def build_mini_court():
    court_draw_w = MINI_W - 2 * padding
    court_draw_h = MINI_H - 2 * padding
    sx = court_draw_w / DOUBLES_W
    sy = court_draw_h / COURT_H

    x_dl = padding;              x_dr = MINI_W - padding
    x_l  = padding + int(ALLEY * sx)
    x_r  = padding + int((DOUBLES_W - ALLEY) * sx)
    cx   = padding + int((DOUBLES_W / 2) * sx)
    top  = padding;              bot  = MINI_H - padding
    y1   = padding + int(6.40 * sy)
    y2   = padding + int((COURT_H - 6.40) * sy)
    ny   = padding + court_draw_h // 2

    base = np.full((MINI_H, MINI_W, 3), 255, dtype=np.uint8)
    cv2.rectangle(base, (2, 2),       (MINI_W-2, MINI_H-2), (0,0,0),     2)
    cv2.rectangle(base, (x_dl, top),  (x_dr, bot),           (0,200,0),   2)
    cv2.line(base, (x_l, top),  (x_l, bot),   (0,200,0), 1)
    cv2.line(base, (x_r, top),  (x_r, bot),   (0,200,0), 1)
    cv2.line(base, (x_l, y1),   (x_r, y1),    (0,200,0), 1)
    cv2.line(base, (x_l, y2),   (x_r, y2),    (0,200,0), 1)
    cv2.line(base, (cx,  y1),   (cx,  y2),    (0,200,0), 1)
    cv2.line(base, (x_dl, ny),  (x_dr, ny),   (0,200,0), 1)
    cv2.line(base, (cx-5, top), (cx+5, top),  (0,200,0), 2)
    cv2.line(base, (cx-5, bot), (cx+5, bot),  (0,200,0), 2)
    return base