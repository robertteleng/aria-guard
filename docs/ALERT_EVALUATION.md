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

## Candidate change 1: metric in-path threat (pre-registered 2026-09-17)

Registered after the baseline results above and **before writing its code**.
Parameters are fixed here and are not tuned on the evaluation recordings.

**Motivation from the baseline.** Pooled alert precision was 13–15 % and
episode recall 15–16 %; with a 5 m corridor (sensitivity) precision was still
~35 %. aria-guard decides "in the path" from image thirds (left / centre /
right) and a relative depth renormalized every frame, so it cannot tell a car
parked 4 m to the side from one ahead.

**Change.** Each detection gets a metric position computed only from data
available live on the glasses:
- the ray through the bottom-centre of the bbox (RGB calibration);
- the gravity direction from the accelerometer, averaged over the last 1.0 s;
- a fixed eye height of **1.6 m** above flat ground;
- forward = the camera's optical axis projected on the horizontal plane.

This gives forward distance *F* and lateral offset *L* in metres. Channel-A
threat becomes:

| Level | Rule |
|---|---|
| DANGER | \|L\| ≤ 0.75 m and 0 < F ≤ 1.5 m |
| WARNING | \|L\| ≤ 0.75 m and 1.5 < F ≤ 3.0 m |
| ATTENTION | \|L\| ≤ 0.75 m and 3.0 < F ≤ 5.0 m |
| NONE | otherwise, or no ground contact |

Tracking, the arbiter (cooldowns, rate limit) and channel B are unchanged.

**Evaluation.** Same six recordings, same detections, same reference and
metrics, compared with `ttc_fix`. The reference uses the SLAM ground plane and
the **future path actually walked**, while the candidate uses gravity, a fixed
eye height and the **current camera heading**. They share the ground-contact
idea, so independence is partial; the agreement between the candidate's *F*
and the reference distance is reported next to the metrics.

**Decision.** The registered rule applies (recall must not drop; unjustified
alerts per minute must not rise more than 10 %, or recall +5 points with at
most +20 %). A candidate that passes also has to show precision up in at least
4 of the 6 recordings, so that a pooled gain is not driven by one recording.

## Result: candidate change 1 (2026-09-17)

Six recordings, same NUC replay detections, records in
`benchmarks/alerts/candidate1/` (commit `417b1b9`). Tables:
`python scripts/alert_table.py benchmarks/alerts/candidate1/*.json --variants ttc_fix metric_inpath`.

| Variant | Channel-A alerts | Alert precision | Hazard episodes | Episode recall | Unjustified alerts/min |
|---|---|---|---|---|---|
| ttc_fix (heuristic) | 370 | 14.6 % (54) | 383 | 15.7 % (60) | 8.57 |
| metric_inpath | 319 | **37.6 %** (120) | 381 | **33.1 %** (126) | **5.40** |

Per recording, metric_inpath raised precision in 6 of 6, raised recall in 6 of
6 and lowered unjustified alerts per minute in 6 of 6.

**Decision: adopted.** It passes the registered rule and the 4-of-6 condition.
The live pipeline uses it when the source provides RGB calibration and IMU
(`ARIA_THREAT_MODEL=heuristic` restores the previous score). Offline evaluation
and the live pipeline share one implementation, `src/input/metric_inpath.py`;
the refactor reproduced the recorded results exactly.

**What this does not settle:**
- **Independence is partial.** The candidate's forward distance agrees only
  loosely with the reference (median absolute difference 0.58–3.00 m per
  recording) and is systematically shorter (median signed −0.39 to −2.94 m),
  consistent with the wearers' eye height differing from the fixed 1.6 m. Both
  use the bbox bottom, so part of the gain may come from that shared choice. A
  live session is the independent check.
- **DANGER dominates.** 197 of the 319 alerts are DANGER, a level that bypasses
  the arbiter's rate limit. Median lead time per recording is 1.1–2.8 s with a
  10th percentile down to 0.08 s: many warnings come late.
- Next candidates, to be registered the same way: per-wearer eye-height
  estimate, a distance-aware DANGER threshold with ego speed, and lead time as a
  registered metric.

## Amendment 2026-09-17: the wearer's own body (error found, before recomputing)

**Error.** While preparing the README clip, the hazard it showed was a `person`
that was the wearer's own hand holding a phone. Twelve random `person` alerts
whose box touches the bottom edge were inspected: all twelve were the wearer's
hand or arm. The reference counted these as objects in the path, so hazard
episodes and "justified" alerts included the wearer's body, and the published
numbers for candidate change 1 (37.6 % precision, 33.1 % recall) are inflated.
They are withdrawn until recomputed under this amendment.

**Reference, corrected.** MPS hand tracking gives each hand's wrist position in
the device frame (`hand_tracking_frames.jsonl`, `T_wrist_device.translation`, mm).
A detection is **the wearer's body** when its class is `person` and the wrist of
a hand with `existence_confidence >= 0.5`, within 60 ms of the frame, projects
inside its box enlarged by 10 % on each side. The convention was checked before
any metric: on `recording_968813465288489`, 92 % of tracked wrists paired with
bottom-edge `person` boxes fall inside them, against 2.6 % for other boxes.
Wearer-body detections get no ground position: they never form hazard
episodes, and an alert about a track with no in-path time is unjustified. This
applies to every variant, including the ones already reported.

**Candidate change 2: wearer-body filter (live-available data only).** Hand
tracking is not available live, so the pipeline uses image geometry and gravity:
a `person` detection is treated as the wearer's body when its box bottom is at
or below 97 % of the image height **and** the ray through its top-centre points
at least 15° below horizontal. A standing person close enough to cut the bottom
edge has a head near or above eye level; a hand or arm held in front does not.
Such detections get threat level `NONE`. Parameters fixed here; 10° and 20° are
reported as sensitivity only.

**Reported.** For `ttc_fix`, `metric_inpath` and `metric_inpath_selfbody`, the
usual metrics under the corrected reference, pooled and per recording. For the
filter against the hand-tracking reference: share of wearer-body detections it
removes, and share of the detections it removes that the reference does not
mark as wearer body (possible real people suppressed).

**Decision rule.** Same as candidate change 1: recall must not drop; unjustified
alerts per minute must not rise more than 10 % (or recall +5 points with at most
+20 %); precision up in at least 4 of 6 recordings. The comparison is
`metric_inpath_selfbody` against `metric_inpath`, both under the corrected
reference. The README is updated with the corrected numbers whatever the result.
