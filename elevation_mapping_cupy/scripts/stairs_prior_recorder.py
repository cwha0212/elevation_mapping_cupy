#!/usr/bin/env python3
"""Register the stairs during MAPPING (prior_record:=true, next to terrain_prior_recorder).

While the operator drives the robot and NAVI maps, this node keeps
  - the robot's 3D trajectory: TF map <- cloud frame (lidar_frame) at every cloud stamp
  - the clouds (/points/merged_deskewed) accumulated in map coordinates as 5 cm voxels within max_range of the robot:
    centroid, point count, number of clouds that saw the voxel, first/last time seen
On save (Trigger ~/save, or at shutdown when save_on_shutdown) it writes <out_dir>/stairs_record.npz (a temporary file,
removed by the fit on success, kept for an offline refit on failure) and starts stairs_fit.py on it in a detached
process (the fit takes ~20 s on the 334 s / 6-flight 2026-10-27 bag, longer than launch waits at shutdown), which writes
  <out_dir>/map_stairs.yaml   flights (map frame: first riser, axis, lateral extent, n, rise, tread, levels) + staircases
  <out_dir>/map_stairs.png    review figure: top view of the registered flights + per-flight section with the fitted steps
The navigation stack loads map_stairs.yaml through stairs_prior_publisher.py (stairs:=true stairs_map:=...).

Why the trajectory and the accumulated cloud and not the live stairs layer: the merged cloud is a 0.15 m voxel grid, a
single frame cannot separate 0.165 m risers (stairs_filter confirmed 0 % of 6 flights on the 2026-10-27 bag); the robot's
own height along its path finds every flight it walked, and many frames in different voxel phases resolve the steps
(6/6 flights, first riser within 2 cm of the offline fit). Flights that were seen but never walked are not registered.

Cost: one np.unique per cloud and a sorted merge every merge_every clouds; ~2-6 M voxels for a 300 s run.
"""
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_srvs.srv import Trigger
import tf2_ros


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class StairsRecorder(Node):
    def __init__(self):
        super().__init__("stairs_prior_recorder")
        g = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.topic = g("cloud_topic", "/points/merged_deskewed")
        self.out_dir = str(g("out_dir", os.path.expanduser("~/terrain_prior")))
        self.map_frame = g("map_frame", "map")
        self.voxel = float(g("voxel", 0.05))
        self.max_range = float(g("max_range", 12.0))
        self.min_range = float(g("min_range", 0.3))
        self.sensor_height = float(g("sensor_height", 0.74))
        self.merge_every = int(g("merge_every", 20))
        self.save_on_shutdown = bool(g("save_on_shutdown", True))
        self.max_tf_age = float(g("max_tf_age", 0.15))
        # memory bound: every prune_every merges drop voxels last seen > keep_recent_s ago that lie farther than keep_r from
        # every stair traversal found so far (the fit only reads +-2 m along / +-2.5 m across a flight). Without it the table
        # grew with the mapped area: 7.0 M voxels / 629 MB RSS for the 334 s 1f bag.
        self.prune_every = int(g("prune_every", 10)); self.keep_recent_s = float(g("keep_recent_s", 60.0)); self.keep_r = float(g("keep_r", 4.0))
        self.n_pruned = 0
        self.run_fit = bool(g("run_fit", True))
        self.tf = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30.0)); # TF on its own node + thread: Humble's spin_thread listener spins the node it is given, so with `self` that thread
        # also ran the cloud callbacks and the buffer fell ~4.5 s behind the clouds (every lookup failed on the 1f bag)
        self.tf_node = rclpy.create_node("stairs_prior_recorder_tf", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", value=bool(self.get_parameter("use_sim_time").value))])
        self.tfl = tf2_ros.TransformListener(self.tf, self.tf_node, spin_thread=True)
        # master voxel table (sorted keys)
        self.keys = np.zeros(0, np.int64); self.sum = np.zeros((0, 3)); self.cnt = np.zeros(0, np.int64)
        self.frames = np.zeros(0, np.int32); self.tfirst = np.zeros(0); self.tlast = np.zeros(0)
        self.batch = []
        self.traj = []
        self.t_cb = self.t_merge = 0.0; self.n_cb = self.n_merge = 0
        self.n = self.n_tf_fail = 0; self.t0 = time.time(); self.saved = False
        self.create_subscription(PointCloud2, self.topic, self.cb, qos_profile_sensor_data)
        self.create_service(Trigger, "~/save", self.on_save)
        from visualization_msgs.msg import MarkerArray
        q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_viz = self.create_publisher(MarkerArray, "~/markers", q)
        self.create_timer(2.0, self.publish_viz)
        # rviz: the accumulated map (voxels seen >= 2 times, at most viz_max_points) every viz_cloud_s
        self.pub_cloud = self.create_publisher(PointCloud2, "~/accumulated", q)
        self.viz_cloud_s = float(g("viz_cloud_s", 3.0)); self.viz_max = int(g("viz_max_points", 400000))
        if self.viz_cloud_s > 0:
            self.create_timer(self.viz_cloud_s, self.publish_cloud)
        self.get_logger().info("recording stairs: %s in %s, %.2f m voxels within %.0f m -> %s/map_stairs.yaml on save" % (
            self.topic, self.map_frame, self.voxel, self.max_range, self.out_dir))

    def cb(self, m):
        tw = time.perf_counter()
        try:
            self._cb(m)
        finally:
            self.t_cb += time.perf_counter() - tw; self.n_cb += 1

    def _cb(self, m):
        st = rclpy.time.Time.from_msg(m.header.stamp)
        ts = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        try:
            tr = self.tf.lookup_transform(self.map_frame, m.header.frame_id, st, timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            tr = None
            try:     # latest transform if it is within max_tf_age of the cloud (TF publishers that tick on their own clock)
                lt = self.tf.lookup_transform(self.map_frame, m.header.frame_id, rclpy.time.Time())
                if abs(lt.header.stamp.sec + lt.header.stamp.nanosec * 1e-9 - ts) <= self.max_tf_age:
                    tr = lt
            except Exception:
                pass
            if tr is None:
                self.n_tf_fail += 1
                if self.n_tf_fail % 50 == 1:
                    self.get_logger().warn("no TF %s <- %s at the cloud stamp (%d clouds skipped): %s" % (
                        self.map_frame, m.header.frame_id, self.n_tf_fail, str(e)[:160]))
                return
        R = quat_to_rot(tr.transform.rotation); p = tr.transform.translation; pv = np.array([p.x, p.y, p.z])
        self.traj.append((ts, p.x, p.y, p.z))
        arr = pc2.read_points(m, field_names=("x", "y", "z"), skip_nans=True)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64) if len(arr) else np.zeros((0, 3))
        r = np.linalg.norm(xyz[:, :2], axis=1)
        xyz = xyz[(r > self.min_range) & (r < self.max_range)]
        if not len(xyz):
            return
        w = xyz @ R.T + pv
        ijk = np.floor(w / self.voxel).astype(np.int64) + (1 << 20)
        key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
        uk, inv, c = np.unique(key, return_inverse=True, return_counts=True)
        s = np.zeros((len(uk), 3)); np.add.at(s, inv, w)
        self.batch.append((uk, s, c, ts))
        self.n += 1; self.saved = False          # a later shutdown saves again after a mid-run ~/save
        if len(self.batch) >= self.merge_every:
            self.merge()
        if self.n % 500 == 0:
            self.get_logger().info("clouds %d, voxels %d (%.0f MB), trajectory z %.2f .. %.2f, %.0f s; cost per cloud %.1f ms "
                                   "(of which merge+prune %.1f ms averaged), pruned %d" % (
                self.n, len(self.keys), self.keys.nbytes * 7.5 / 1e6, min(q[3] for q in self.traj), max(q[3] for q in self.traj),
                time.time() - self.t0, 1e3 * self.t_cb / max(self.n_cb, 1), 1e3 * self.t_merge / max(self.n_cb, 1), self.n_pruned))

    def publish_cloud(self):
        if not len(self.keys) or self.pub_cloud.get_subscription_count() == 0:
            return
        k = np.flatnonzero(self.cnt >= 2)
        if len(k) > self.viz_max:
            k = k[np.linspace(0, len(k) - 1, self.viz_max).astype(np.int64)]
        P = (self.sum[k] / self.cnt[k, None]).astype(np.float32)
        from std_msgs.msg import Header
        h = Header(); h.frame_id = self.map_frame; h.stamp = self.get_clock().now().to_msg()
        self.pub_cloud.publish(pc2.create_cloud_xyz32(h, P))

    def publish_viz(self):
        """rviz: trajectory (foot height, red where the walked slope says 'stairs') + the stair traversals found so far."""
        if len(self.traj) < 30:
            return
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray
        self._stairs_fit()
        tr = np.array(self.traj); t = tr[:, 0] - tr[0, 0]; foot = tr[:, 1:4].copy(); foot[:, 2] -= self.sensor_height
        trav, zf, slope = self._sf.find_flights(t, foot)
        now = self.get_clock().now().to_msg(); ma = MarkerArray()
        line = Marker(); line.header.frame_id = self.map_frame; line.header.stamp = now; line.ns = "trajectory"; line.id = 0
        line.type = Marker.LINE_STRIP; line.scale.x = 0.06; line.pose.orientation.w = 1.0
        step = max(1, len(tr) // 3000)
        for i in range(0, len(tr), step):
            line.points.append(Point(x=float(foot[i, 0]), y=float(foot[i, 1]), z=float(zf[i])))
            st = min(1.0, abs(slope[i]) / 0.4)
            line.colors.append(ColorRGBA(r=float(st), g=float(1 - st), b=0.2, a=1.0))
        ma.markers.append(line)
        for k, d in enumerate(trav):
            a = Marker(); a.header = line.header; a.ns = "stair traversals"; a.id = k; a.type = Marker.ARROW
            a.scale.x = 0.12; a.scale.y = 0.25; a.scale.z = 0.25; a.color = ColorRGBA(r=1.0, g=0.2, b=0.1, a=0.9)
            i0, i1 = (d["i0"], d["i1"]) if d["up"] else (d["i1"], d["i0"])
            a.points = [Point(x=float(foot[i0, 0]), y=float(foot[i0, 1]), z=float(zf[i0]) + 0.3),
                        Point(x=float(foot[i1, 0]), y=float(foot[i1, 1]), z=float(zf[i1]) + 0.3)]
            ma.markers.append(a)
        txt = Marker(); txt.header = line.header; txt.ns = "status"; txt.id = 0; txt.type = Marker.TEXT_VIEW_FACING
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(foot[-1, 0]), float(foot[-1, 1]), float(zf[-1]) + 1.6
        txt.pose.orientation.w = 1.0; txt.scale.z = 0.3; txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        txt.text = "stairs recorder: %d stair traversals, %d clouds, %d voxels" % (len(trav), self.n, len(self.keys))
        ma.markers.append(txt)
        self.pub_viz.publish(ma)

    def merge(self):
        if not self.batch:
            return
        tw = time.perf_counter()
        try:
            self._merge()
        finally:
            self.t_merge += time.perf_counter() - tw; self.n_merge += 1

    def prune(self):
        if not len(self.keys) or len(self.traj) < 30:
            return
        tr = np.array(self.traj); now = tr[-1, 0]
        keep = self.tlast >= now - self.keep_recent_s
        sf = self._stairs_fit()
        foot = tr[:, 1:4].copy(); foot[:, 2] -= self.sensor_height
        trav, _, _ = sf.find_flights(tr[:, 0] - tr[0, 0], foot)
        if trav:
            from scipy.spatial import cKDTree
            idx = np.concatenate([np.arange(d["i0"], d["i1"] + 1) for d in trav])
            tree = cKDTree(foot[idx, :2])
            old = np.flatnonzero(~keep)
            if len(old):
                c = self.sum[old, :2] / self.cnt[old, None]
                dd, _ = tree.query(c, distance_upper_bound=self.keep_r)
                keep[old[np.isfinite(dd)]] = True
        n0 = len(self.keys)
        if keep.all():
            return
        self.keys, self.sum, self.cnt = self.keys[keep], self.sum[keep], self.cnt[keep]
        self.frames, self.tfirst, self.tlast = self.frames[keep], self.tfirst[keep], self.tlast[keep]
        self.n_pruned += n0 - len(self.keys)

    def _stairs_fit(self):
        if not hasattr(self, "_sf"):
            from haechi_terrain import stairs_fit
            self._sf = stairs_fit
        return self._sf

    def _merge(self):
        K = np.concatenate([b[0] for b in self.batch]); S = np.concatenate([b[1] for b in self.batch])
        C = np.concatenate([b[2] for b in self.batch]); T = np.concatenate([np.full(len(b[0]), b[3]) for b in self.batch])
        self.batch = []
        uk, inv = np.unique(K, return_inverse=True)
        s = np.zeros((len(uk), 3)); np.add.at(s, inv, S)
        c = np.bincount(inv, weights=C).astype(np.int64); fr = np.bincount(inv).astype(np.int32)   # one entry per cloud per voxel
        tf_ = np.full(len(uk), np.inf); np.minimum.at(tf_, inv, T); tl_ = np.full(len(uk), -np.inf); np.maximum.at(tl_, inv, T)
        idx = np.searchsorted(self.keys, uk)
        found = (idx < len(self.keys)) & (self.keys[np.minimum(idx, len(self.keys) - 1)] == uk) if len(self.keys) else np.zeros(len(uk), bool)
        fi = idx[found]
        self.sum[fi] += s[found]; self.cnt[fi] += c[found]; self.frames[fi] += fr[found]
        self.tfirst[fi] = np.minimum(self.tfirst[fi], tf_[found]); self.tlast[fi] = np.maximum(self.tlast[fi], tl_[found])
        new = ~found
        if new.any():
            pos = idx[new]
            self.keys = np.insert(self.keys, pos, uk[new]); self.sum = np.insert(self.sum, pos, s[new], axis=0)
            self.cnt = np.insert(self.cnt, pos, c[new]); self.frames = np.insert(self.frames, pos, fr[new])
            self.tfirst = np.insert(self.tfirst, pos, tf_[new]); self.tlast = np.insert(self.tlast, pos, tl_[new])
        if self.prune_every > 0 and (self.n_merge + 1) % self.prune_every == 0:
            self.prune()

    def save(self):
        self.merge()
        if len(self.traj) < 20 or not len(self.keys):
            self.get_logger().warn("nothing recorded (%d poses), no stairs written" % len(self.traj)); return False, "nothing recorded"
        os.makedirs(self.out_dir, exist_ok=True)
        tr = np.array(self.traj); t0 = tr[0, 0]
        keep = self.cnt >= 2
        path = os.path.join(self.out_dir, "stairs_record.npz")
        np.savez(path, t=tr[:, 0] - t0, t0=t0, lidar=tr[:, 1:4], sensor_height=self.sensor_height,
                 points=(self.sum[keep] / self.cnt[keep, None]).astype(np.float32), count=self.cnt[keep].astype(np.uint32),
                 frames=self.frames[keep], t_first=self.tfirst[keep] - t0, t_last=self.tlast[keep] - t0)
        msg = "stairs record saved: %s (%d poses, %d voxels, z %.2f .. %.2f)" % (path, len(tr), int(keep.sum()), tr[:, 3].min(), tr[:, 3].max())
        self.get_logger().info(msg); self.saved = True
        if self.run_fit:
            # the fit writes stairs_fit.log itself (same run as the yaml); stairs_fit.err only catches a crash. The record
            # is a temporary file: the fit removes it when it succeeds and keeps it when it fails (refit offline).
            fit = os.path.join(os.path.dirname(os.path.realpath(__file__)), "stairs_fit.py")
            err = open(os.path.join(self.out_dir, "stairs_fit.err"), "w")
            subprocess.Popen([sys.executable, fit, "--record", path, "--out", self.out_dir, "--delete_record"], stdout=subprocess.DEVNULL,
                             stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
            msg += "; stairs_fit.py started (map_stairs.yaml/png, log stairs_fit.log; the record is removed on success)"
        return True, msg

    def on_save(self, req, res):
        res.success, res.message = self.save(); return res


def main():
    rclpy.init(); n = StairsRecorder()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    except Exception as e:                  # noqa: BLE001 - ExternalShutdownException on SIGINT; anything else is a bug: say so
        if type(e).__name__ != "ExternalShutdownException":
            import traceback
            n.get_logger().error("stopped by %s:\n%s" % (e, traceback.format_exc()))
    finally:
        if n.save_on_shutdown and not n.saved:
            try:
                n.save()
            except Exception as e:      # noqa: BLE001
                print("stairs_prior_recorder: save failed: %s" % e)
    try:
        n.destroy_node(); rclpy.shutdown()
    except BaseException:
        pass


if __name__ == "__main__":
    main()
