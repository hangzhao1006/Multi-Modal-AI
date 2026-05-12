# Speaker Script — PG-MoE Analysis

**Total: ~1 min 40 sec across 4 slides**

Bold = emphasize. Read at natural pace. Period = small pause. Paragraph break = pause + look at slide.

---

## Slide 1 — Analysis Vision  (~22 sec)

Let me start with the vision side. We use ResNet3D pretrained on Kinetics. It gets us to **89.3%** on its own.

But Grad-CAM tells us to be careful. The model focuses on the person, but also on the background edges. So there's some **scene bias** creeping in. That's a generalization risk, and a clear reason to bring in IMU.

The t-SNE on the bottom shows the feature space is otherwise clean. All 27 classes form well-separated clusters. So vision is strong, but with this background leak that IMU should help fix.

---

## Slide 2 — Analysis: IMU Expert  (~20 sec)

On the IMU side, our deep 1D-CNN hits **82.33%**. That's **14 points up** from midterm.

The per-class story is interesting. IMU nails anything periodic. Walking, squat, drawing circles. 13 classes already at **100%**. But it falls off a cliff on impact actions. Throw is **25%**, knock **44%**, tennis serve **50%**. A wrist sensor just can't tell which way the ball went.

So IMU has a clear blind spot. And conveniently, vision is strong exactly there. The two modalities are complementary.

---

## Slide 3 — Analysis: Phase-Aware Arbitrator  (~28 sec)

That brings us to the arbitrator. It picks which modality to trust at each moment, using only physics from the IMU.

We pull three features. Acceleration, jerk, energy rate. They go into a tiny MLP, just **2.8K parameters**.

The left chart shows why this can work. Every action has its own physics signature.

The real proof is on the right. **Before training, alpha is flat at 0.5. After training, the curves split apart.** Each action takes its own path. So the model actually learns when to trust what. We didn't hardcode any of it.

---

## Slide 4 — Looking back + What's next  (~28 sec)

So zooming out, two things stand out.

First, both modalities are genuinely complementary. Each has a clear blind spot the other fills. The **Oracle ceiling at 98.4%** tells us there's real headroom any reasonable fusion could chase.

Second, an honest finding. **Simple late fusion actually beat our feature-level gating** here. With only 430 training samples, the more complex gating may just be more than the data supports.

That points pretty directly to what's next. **Bigger datasets** so the arbitrator has more to learn from. **Skeleton as a third modality** toward that 98% ceiling. And real-world **egocentric** setups.

Everything's on GitHub. Thanks.
