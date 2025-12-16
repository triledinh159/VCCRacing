import cv2
import numpy as np
from collect_data import AVClient, extract_real_road

# ================= CONFIG =================
IMG_SIZE = (160, 120)

MAX_STEER = 25
WHEELBASE = 2.7

V_MAX = 40
V_MIN = 25
# ==========================================

def compute_target_point(mask, speed):
    h, w = mask.shape
    cx = w // 2

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    mid_y = int(h * 0.7)
    row = dist[mid_y]
    if row.max() < 5:
        return None, 1.0

    road_width = row.max()
    curvature = np.clip(1.0 / (road_width + 1e-3), 0.0, 1.0)

    Ld = 18 + 0.25 * speed - 12 * curvature
    Ld = int(np.clip(Ld, 12, h - 10))
    y = h - Ld

    row = dist[y]
    if row.max() < 5:
        return None, curvature

    tx = int(np.argmax(row))
    ty = y

    return (tx, ty), curvature


def compute_speed(curvature, steer):
    speed = 40 - 18 * curvature - 0.6 * abs(steer)
    return int(np.clip(speed, 18, 40))


def pure_pursuit_steer(tx, ty):
    h = IMG_SIZE[1]
    cx = IMG_SIZE[0] // 2

    dx = tx - cx
    dy = h - ty
    if dy <= 0:
        return 0.0

    curvature = 2 * dx / (dx**2 + dy**2 + 1e-6)
    steer = np.degrees(np.arctan(WHEELBASE * curvature * 1.5))

    return float(np.clip(steer, -MAX_STEER, MAX_STEER))


# ---------- SAFETY GUARD (VERY LIGHT) ----------
def lateral_guard(mask, steer):
    h, w = mask.shape
    cx = w // 2

    y = int(h * 0.9)
    xs = np.where(mask[y] > 0)[0]
    if len(xs) < 20:
        return steer

    road_cx = (xs.min() + xs.max()) / 2
    offset = road_cx - cx
    norm = offset / cx

    if abs(norm) > 0.35:
        guard = np.degrees(np.arctan2(offset, 18))
        return 0.7 * steer + 0.3 * guard

    return steer


def main():
    with AVClient() as client:
        print("[INFO] Pure Pursuit driving started (smooth + safe)")

        while True:
            raw = client.get_raw_image()
            seg = client.get_segmented_image()

            mask, _ = extract_real_road(raw, seg)
            mask = cv2.resize(mask, IMG_SIZE)

            speed = V_MAX

            tgt, curvature = compute_target_point(mask, speed)
            if tgt is None:
                client.set_control(speed=0, angle=0)
                client.get_state_data()
                continue

            tx, ty = tgt
            steer = pure_pursuit_steer(tx, ty)

            # Safety guard (only near edge)
            steer = lateral_guard(mask, steer)

            speed = compute_speed(curvature, steer)

            if speed > 32:
                steer = np.clip(steer, -15, 15)

            client.set_control(speed=int(speed), angle=int(round(steer)))
            client.get_state_data()

            print(f"[PURE] speed={speed:.1f} steer={steer:.2f}")

            vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (tx, ty), 4, (0, 0, 255), -1)
            cv2.imshow("Pure Pursuit (Smooth)", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
