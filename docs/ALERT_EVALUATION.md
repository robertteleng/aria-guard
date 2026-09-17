# Alert evaluation against the wearer's real path (pre-registered)

Written and committed **before** any number was computed. Changing a threshold
after seeing results requires a new dated section explaining why; the original
definitions stay here.

## Question

The replay benchmark shows how much aria-guard alerts (9–11 per minute on the
six outdoor recordings), not whether those alerts were deserved. This
evaluation answers: **when an obstacle lay in the path the person actually
walked, did aria-guard warn in time, and how many alerts were about objects
that never were in that path?**

## Automatic reference (no manual labels)

Each Reading in the Wild recording ships Meta's Machine Perception Services
(MPS) output:
- `slam/closed_loop_trajectory.csv`: the glasses' pose at 1 kHz in a metric,
  gravity-aligned world frame.
- `slam/semidense_points.csv.gz`: static 3D points with uncertainty.
- `eye_gaze/general_eye_gaze.csv`: offline eye gaze.

For every detection the replay produced (same detections file as the alert
metrics), the reference computes:

1. **Metric position by ground contact.** The ray through the bottom-centre of
   the bbox, unprojected with the RGB camera calibration of the recording and
   posed with the MPS trajectory at the frame timestamp, intersects the ground
   plane. The ground height is the 5th percentile of the z of semidense points
   within 5 m (horizontally) of the device, per frame; if fewer than 200
   points qualify, the frame is skipped and counted. Detections whose ray does
   not hit the ground in front of the camera within 15 m are marked
   `no_ground_contact` (for example traffic signs and street lights seen from
   below) and excluded from hazard episodes.
2. **Cross-check for static objects.** For classes that do not move (car,
   truck, bus, bench, fire hydrant, stop sign, traffic light, potted plant,
   Door, Stairs, Street light, Traffic sign, Tree), the median distance of the
   semidense points with `dist_std` ≤ 0.2 m that project inside the central
   half of the bbox is compared with the ground-contact distance. The
   agreement (median absolute difference and share within 25 %) is published
   as the error bar of the reference. Parked cars can move; the class list is
   an approximation stated as such.
3. **Path corridor.** The wearer's future path is the device trajectory over
   the next **3.0 s**, projected on the ground plane.

## Definitions

- **In-path**: an object whose ground position lies within **0.75 m**
  (horizontal) of the future path polyline, at a point of the path reached
  within 3.0 s; or that is already within **1.0 m** of the device and within
  ±30° of the walking direction.
- **Hazard episode**: consecutive frames in which the same tracked object is
  in-path, merged when interrupted by less than 0.5 s. Its *closest approach*
  is the time of minimum distance to the path.
- **Justified alert**: a channel-A alert (ATTENTION, WARNING or DANGER) whose
  object, matched to a track by IoU ≥ 0.5 at the alert frame, is in-path at
  some time within **[t, t + 3.0 s]**.
- **Warned episode**: an episode with a channel-A alert about its object
  between **3.0 s before its first in-path frame** and its closest approach.
- **Lead time**: closest approach time minus the first alert time about that
  object, for warned episodes.

## Metrics (per recording and pooled over the six)

| Metric | Definition |
|---|---|
| Alert precision | justified alerts / channel-A alerts |
| Episode recall | warned episodes / hazard episodes |
| Lead time | median and 10th percentile, seconds |
| Unjustified alerts per minute | channel-A alerts that are not justified, per minute of recording |
| Precision and recall by level | the same, split by ATTENTION / WARNING / DANGER |

All metrics are computed for the tracker **before and after the TTC fix**
(commits `3992e54` and `3c56e50`) on the same detections.

## Decision rules fixed now

- A tracker or arbiter change is kept only if, pooled over the six recordings,
  **episode recall does not drop** and **unjustified alerts per minute do not
  rise by more than 10 %**, or if recall rises by at least 5 points while
  unjustified alerts per minute rise by no more than 20 %.
- The evaluation is reported even if it makes aria-guard look bad.

## Known limits

- The wearer is sighted and steers around obstacles, so "in the path actually
  walked" measures proximity to the chosen route, not avoided collisions. An
  object the wearer swerved around early may never enter the 0.75 m corridor.
- Ground contact assumes locally flat ground; stairs and curbs break it near
  the edge. Detections on stairs are reported separately.
- Detections are aria-guard's own: an obstacle the detector never saw is not
  an episode here, so recall is recall of the alerting logic given the
  detector, not of the whole system.
- Moving objects (people, bicycles) have no semidense cross-check; their
  distance relies on ground contact alone.

## Amendment 2026-09-17, before computing any result

Found while reading the MPS files, not from results:

1. **Ground height uses only converged points.** `semidense_points.csv.gz`
   contains unconverged points thousands of metres away (e.g. a point at
   z = 281 m with `dist_std` = 5,712 m). The ground height is therefore the 5th
   percentile of z over points with `dist_std` ≤ 0.2 m within 5 m of the
   device, the same filter as the cross-check.
2. **Cross-check sampling and projection.** The cross-check runs on every 10th
   frame that has a static-class detection, which bounds the cost; it
   estimates the error of the reference, it does not feed the metrics. A point
   counts as inside the central half of a bbox when its normalized camera
   coordinates (x/z, y/z) fall inside those of the four unprojected corners of
   that central half, an approximation of the fisheye footprint that is tight
   for regions this small.
3. **Timestamps.** Detection frames map to the VRS RGB capture time (device
   clock); MPS `tracking_timestamp_us` uses the same clock.

## Amendment 2026-09-17, after inspecting the cross-check (alert metrics unchanged)

The cross-check disagreed strongly on some recordings (median absolute
difference up to 4.5 m). Inspection before any decision on alert logic: in
those recordings the static detections sampled were almost all **Tree**, and
the semidense points inside the central half of a tree bbox are canopy and
background behind it, so the cross-check distance was about 3 m **longer**
than the ground contact at the trunk base (ground − semidense median −3.1 m).
Eye height above the estimated ground stayed plausible (median 1.48 m and
1.80 m on the two recordings checked), so the ground plane was not the cause.

Therefore:
1. The cross-check is reported **per class**, and its rows are stored in the
   record so it can be re-summarized without recomputation.
2. The headline agreement **excludes Tree**, whose bbox centre is canopy or
   background by construction. Tree agreement is still reported.
3. This changes only how the reference's error is reported. Hazard episodes,
   alert matching and every alert metric are computed exactly as registered.
