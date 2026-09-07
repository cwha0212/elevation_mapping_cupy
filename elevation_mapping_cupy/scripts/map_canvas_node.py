#!/usr/bin/env python3
"""Re-canvas the accumulated octomap onto a fixed-extent unknown map.

Two cold-start facts make the raw octomap projection unusable as Nav2's
static map. Its extent is the explored bounding box, so any goal past the
frontier is "off the global costmap" and planning fails outright -- the exact
opposite of "start empty, drive to a coordinate". And until the first point
is inserted it publishes nothing at all, which leaves StaticLayer waiting
and the planner refusing to start while the robot waits for a plan to move.

A fixed all-unknown canvas solves both at once: it is published from the
first tick (StaticLayer satisfied, allow_unknown routes through the unknown),
and every octomap update is pasted onto it, so known space grows inside a
frame whose corners never move and any goal within the canvas stays
plannable.

Publishes transient-local, so late subscribers (RViz included) get the
current state immediately.
"""
import numpy as np
import rclpy
from scipy import ndimage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Bool


class MapCanvasNode(Node):
    def __init__(self) -> None:
        super().__init__("map_canvas_node")
        self.frame = self.declare_parameter("map_frame", "odom").value
        self.size = float(self.declare_parameter("size", 40.0).value)
        self.resolution = float(self.declare_parameter("resolution", 0.05).value)
        # Unknown is published as a soft cost, not -1: with the static
        # layer set trinary_costmap:false this makes unexplored ground
        # expensive-but-passable, so the planner hugs confirmed-free
        # corridors and crosses unknown only when there is no alternative.
        # That single property is what keeps a cold-start plan from slicing
        # through the unobserved roadway. 0 disables (plain -1 unknown).
        self.unknown_value = int(self.declare_parameter("unknown_value", 70).value)
        # Memory decay: an occupied cell the LIVE sensors have not
        # re-confirmed for decay_sec is demoted back to unknown, and the
        # octree's stale echo of it is suppressed until live evidence marks
        # it again. This is what retires ghosts the robot can never re-view
        # (the wedge behind the box); demotion is to unknown, never to free,
        # so nothing unseen is ever declared safe. 0 disables.
        self.decay_sec = float(self.declare_parameter("decay_sec", 30.0).value)
        self.n = int(round(self.size / self.resolution))
        fill = self.unknown_value if self.unknown_value > 0 else -1
        self.unknown_fill = fill
        self.canvas = np.full((self.n, self.n), fill, dtype=np.int8)
        self.occ_stamp = np.zeros((self.n, self.n), dtype=np.float64)
        self.suppress = np.zeros((self.n, self.n), dtype=bool)

        # Live overlay: octomap is memory, and memory must not outvote the
        # sensors. Wherever the CURRENT elevation window has an answer, that
        # answer overrides the octree at publish time -- a walker's ghost
        # vanishes the moment the lidar re-reads the floor, and a fresh
        # hazard shows up before the octree has accumulated it. Applied at
        # publish only, never baked into the memory canvas, so the two
        # writers cannot ping-pong.
        # 0.55, not the march's free cut: a walker leaves a few-cm mound
        # that scores 0.5-0.65 for a while -- clearly not an obstacle, but
        # below 0.7 the overlay could not overrule the octree's stale
        # occupied mark and the ghost stayed visible. "Not a hazard right
        # now" is exactly the evidence needed to retire a memory.
        self.live_free = float(self.declare_parameter("live_free_threshold", 0.55).value)
        self.live_obstacle = float(self.declare_parameter("live_obstacle_threshold", 0.45).value)
        self.safety_layer = self.declare_parameter("safety_layer", "safety").value
        self._live = None

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, "/projected_map", qos)
        # The local costmap consumes this directly instead of accumulating
        # marks: it is the elevation window's verdict of THIS instant, so a
        # walker's wake cannot linger there by construction. Unknown is
        # published free -- locally, only known dangers matter; the global
        # map carries the caution about the unseen.
        self.live_pub = self.create_publisher(OccupancyGrid, "/live_obstacles", 5)
        # One-shot camera gate: while the camera says the ground straight
        # ahead is bad, a small box in front of the robot is marked in the
        # LIVE grid only. Nothing is written to the global map, and the
        # mark exists exactly as long as the verdict does.
        self._fwd_blocked = False
        self._yaw = 0.0
        self._rx = self._ry = 0.0
        self.create_subscription(Bool, "/front_cam/forward_blocked",
                                 lambda m: setattr(self, "_fwd_blocked", m.data), 5)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(OccupancyGrid, "projected_map_local", self._on_map, 5)
        self.create_subscription(
            GridMap, "/elevation_mapping_node/elevation_map_terrain",
            self._on_terrain, 2,
        )
        # First publish happens before any octomap content exists; afterwards
        # the timer only fires between octomap updates as a keep-current.
        self.create_timer(2.0, self._publish)
        self.get_logger().info(
            f"Canvas {self.size:.0f} m @ {self.resolution} m on '/projected_map'."
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        if abs(msg.info.resolution - self.resolution) > 1e-6:
            self.get_logger().error(
                f"octomap resolution {msg.info.resolution} != canvas "
                f"{self.resolution}; set them equal.", throttle_duration_sec=10.0
            )
            return
        data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width
        )
        c0 = int(round((msg.info.origin.position.x + self.size / 2) / self.resolution))
        r0 = int(round((msg.info.origin.position.y + self.size / 2) / self.resolution))
        r1, c1 = r0 + data.shape[0], c0 + data.shape[1]
        if r0 < 0 or c0 < 0 or r1 > self.n or c1 > self.n:
            self.get_logger().warning(
                "octomap outgrew the canvas; enlarge 'size'.",
                throttle_duration_sec=30.0,
            )
            sr0, sc0 = max(-r0, 0), max(-c0, 0)
            r0, c0 = max(r0, 0), max(c0, 0)
            data = data[sr0: sr0 + (min(r1, self.n) - r0),
                        sc0: sc0 + (min(c1, self.n) - c0)]
            r1, c1 = r0 + data.shape[0], c0 + data.shape[1]
        # Known beats unknown -- but a suppressed (decayed) occupied mark is
        # the octree echoing a memory the live sensors failed to re-confirm;
        # only live evidence may resurrect those cells. Free evidence is
        # always welcome and lifts suppression.
        known = data >= 0
        occ_in = known & (data > 50)
        free_in = known & ~occ_in
        block_c = self.canvas[r0:r1, c0:c1]
        block_sup = self.suppress[r0:r1, c0:c1]
        block_stamp = self.occ_stamp[r0:r1, c0:c1]
        now = self.get_clock().now().nanoseconds * 1e-9
        newly = occ_in & ~block_sup & (block_c <= 50)
        block_stamp[newly] = now
        apply_occ = occ_in & ~block_sup
        block_c[apply_occ] = data[apply_occ]
        block_c[free_in] = data[free_in]
        block_sup[free_in] = False
        self.canvas[r0:r1, c0:c1] = block_c
        self.suppress[r0:r1, c0:c1] = block_sup
        self.occ_stamp[r0:r1, c0:c1] = block_stamp
        self._publish()

    def _on_terrain(self, msg: GridMap) -> None:
        names = list(msg.layers)
        if self.safety_layer not in names or "drivability" not in names:
            return

        def layer(name):
            d = msg.data[names.index(name)]
            h, w = d.layout.dim[0].size, d.layout.dim[1].size
            return np.array(d.data, dtype=np.float32).reshape(h, w)

        # Occupied comes from GEOMETRY alone. The semantic layer is an EMA
        # that remembers its last verdict wherever the camera stopped
        # looking, so letting it mint obstacles turns stale camera memories
        # into ever-growing lethal cells. It may only BLOCK the free
        # promotion (safety gate below); forbidding is cheap, endorsing
        # needs current evidence.
        self._live = (layer(self.safety_layer), layer("drivability"), msg.info)
        self._publish_live()
        self._publish()

    def _on_odom(self, msg) -> None:
        import math
        p = msg.pose.pose
        self._rx, self._ry = p.position.x, p.position.y
        q = p.orientation
        self._yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y * q.y + q.z * q.z))

    def _publish_live(self) -> None:
        _, driv, info = self._live
        arr = driv
        occ = np.isfinite(arr) & (arr < self.live_obstacle)
        # single-cell flickers of a walker's wake are noise, not obstacles
        labels, nl = ndimage.label(occ)
        if nl:
            sizes = ndimage.sum(occ, labels, np.arange(1, nl + 1))
            occ &= ~np.isin(labels, np.nonzero(sizes < 4)[0] + 1)
        self._live_occ = occ
        h, w = arr.shape
        grid = OccupancyGrid()
        grid.header.frame_id = self.frame
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = info.resolution
        grid.info.width = w
        grid.info.height = h
        grid.info.origin.position.x = info.pose.position.x - w / 2 * info.resolution
        grid.info.origin.position.y = info.pose.position.y - h / 2 * info.resolution
        grid.info.origin.orientation.w = 1.0
        data = np.zeros((h, w), dtype=np.int8)
        data[occ[::-1, ::-1]] = 100
        if self._fwd_blocked:
            import math
            res = info.resolution
            ox = info.pose.position.x - w / 2 * res
            oy = info.pose.position.y - h / 2 * res
            c, s_ = math.cos(self._yaw), math.sin(self._yaw)
            for fwd in np.arange(0.5, 1.7, res):
                for lat in np.arange(-0.4, 0.4, res):
                    x = self._rx + fwd * c - lat * s_
                    y = self._ry + fwd * s_ + lat * c
                    ci = int((x - ox) / res); ri = int((y - oy) / res)
                    if 0 <= ri < h and 0 <= ci < w:
                        data[ri, ci] = 100
        grid.data = data.reshape(-1).tolist()
        self.live_pub.publish(grid)

    def _overlay(self, out: np.ndarray) -> None:
        """Write the live window's verdicts over the memory copy."""
        if self._live is None:
            return
        safety, driv, info = self._live
        h, w = safety.shape
        res = info.resolution
        # grid_map convention: row along -Y, col along -X from the centre
        cols = np.arange(w)
        rows = np.arange(h)
        wx = info.pose.position.x - (cols - w / 2 + 0.5) * res
        wy = info.pose.position.y - (rows - h / 2 + 0.5) * res
        cxi = np.round((wx + self.size / 2) / self.resolution).astype(np.int32)
        ryi = np.round((wy + self.size / 2) / self.resolution).astype(np.int32)
        ok_c = (cxi >= 0) & (cxi < self.n)
        ok_r = (ryi >= 0) & (ryi < self.n)
        sub = safety[np.ix_(ok_r, ok_c)]
        sub_d = driv[np.ix_(ok_r, ok_c)]
        tgt = np.ix_(ryi[ok_r], cxi[ok_c])
        free = np.isfinite(sub) & (sub >= self.live_free)
        occ = np.isfinite(sub_d) & (sub_d < self.live_obstacle)
        # despeckle the bake: same blob rule as the live grid
        labels, nl = ndimage.label(occ)
        if nl:
            sizes = ndimage.sum(occ, labels, np.arange(1, nl + 1))
            occ &= ~np.isin(labels, np.nonzero(sizes < 4)[0] + 1)
        block = out[tgt]
        block[free] = 0
        block[occ] = 100
        out[tgt] = block
        # live evidence is the only currency that renews or lifts decay
        now = self.get_clock().now().nanoseconds * 1e-9
        st = self.occ_stamp[tgt]
        st[occ] = now
        self.occ_stamp[tgt] = st
        sup = self.suppress[tgt]
        sup[occ] = False
        sup[free] = False
        self.suppress[tgt] = sup

    def _publish(self) -> None:
        grid = OccupancyGrid()
        grid.header.frame_id = self.frame
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.resolution
        grid.info.width = self.n
        grid.info.height = self.n
        grid.info.origin.position.x = -self.size / 2
        grid.info.origin.position.y = -self.size / 2
        grid.info.origin.orientation.w = 1.0
        if self.decay_sec > 0:
            now = self.get_clock().now().nanoseconds * 1e-9
            expired = (self.canvas > 50) & (now - self.occ_stamp > self.decay_sec)
            if expired.any():
                self.canvas[expired] = self.unknown_fill
                self.suppress[expired] = True
        out = self.canvas.copy()
        self._overlay(out)
        grid.data = out.reshape(-1).tolist()
        self.pub.publish(grid)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapCanvasNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
