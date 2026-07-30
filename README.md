# Personalized Debiasing for Sequential Recommendation

Sequential recommenders like GRU4Rec and SASRec are good at predicting what you'll watch next, and they're also good at learning to recommend whatever is globally popular, to everyone. The usual fix is Inverse Propensity Scoring, which down-weights popular items during training. But IPS applies the *same* correction to every user, and that doesn't match how people actually behave. Someone who watches films across eight genres and someone who only watches Marvel movies get identical treatment.

This project asks a simple question: what if the debiasing strength were personalized?

We give each user two scores. An **explorer score α(u)**, from the entropy of their genre distribution, controls how hard we correct for popularity. A **trend-follower score β(u)**, from the average "trendiness" of the items they've actually interacted with, controls how hard we correct for temporal trending bias. Both feed into a reweighted loss. Nothing about the model architecture changes, and because the correction happens during training, there's **no extra cost at inference**. The debiasing is already incorporated into the learned weights.

We also built a variant where a small **MLP** learns α and β end-to-end instead of computing them from a formula.

Course project at IIIT Bangalore, 2026. Full write-up in [`docs/Final_Report.pdf`](docs/Final_Report.pdf); slides in [`docs/FinalPresentation.pdf`](docs/FinalPresentation.pdf).

---

## The two biases

**Popularity bias** is the static one. A handful of items contributes to most of the interactions in the training data, so the model learns that recommending them is a low-risk bet. Abdollahpouri et al. showed this hurts unevenly: "Niche" users, whose profiles are more than half non-popular items, still get recommendations that are almost entirely popular. That asymmetry is the whole motivation for personalizing the correction.

**Temporal trending bias** is the moving one. Item popularity isn't fixed: something trendy last month may be dead today. If recent training data is dominated by a few trending items, the model pushes them at everyone, including users who reliably prefer older, stable content. We treat this as a separate axis, because an item can be globally popular but not currently trending (a classic film), or trending but not globally popular (a new indie release).

## Method

**Exposure.** For each item we precompute a static popularity exposure `e_pop(i)`, Its share of all training interactions and a time-local trending exposure `e_trend(i, t)`, its share of interactions inside a sliding window of `W` days (default 30). Both are computed once, before training.

**User identity scores.**

```
α(u) = H(u) / log(K)          normalized Shannon entropy over the user's genre distribution
β(u) = mean of e_trend(i, t)  averaged over the items in the user's history
```

α near 1 means the user spreads across genres; α near 0 means they live in one. β near 1 means they consume whatever's trendy at the time; β near 0 means their taste is time-stable. Both land in [0, 1] without extra normalization.

**The loss.** Standard cross-entropy, reweighted:

```
L = [ 1 / ( e_pop(i)^α(u) · e_trend(i,t)^(1−β(u)) ) ] · ( −log P(i) )
```

High α means rare items get a large boost for that user. Low β means the exponent `1−β` is large, so non-trending items get pushed up for someone with stable taste. Weights are clipped at 5.0 so rare items can't blow up the gradients.

**Learned variant.** The formula has blind spots, high entropy computed from five interactions isn't the same evidence as high entropy from five hundred. So we also train a small MLP that takes four user features (entropy, average popularity of their history, normalized profile size, trending correlation) and outputs α and β through a sigmoid. It gets no direct supervision; it's trained jointly with the recommender by the same optimizer, and just learns whatever debiasing strengths minimize the recommendation loss. At inference it's discarded.

## Setup

MovieLens-1M — roughly 1M ratings, 6K users, 4K movies, 18 genres. Ratings ≥ 4 count as positive implicit feedback; users with fewer than 5 interactions are dropped.

**Split: per-user temporal holdout.** For each user, the last interaction is test, the second-to-last is validation, everything before is train. Strictly ordered, no future leakage.

GRU4Rec backbone: 128-d embeddings, 256 hidden, 1 layer, dropout 0.4, max sequence 50. Adam at lr 1e-3, weight decay 1e-4, batch 512, up to 50 epochs with patience 10. Trained on GPU with AMP.

## Results

Evaluated at K = 10.

| Method | Recall@10 | NDCG@10 | Coverage | Long-tail | ΔGAP Niche |
|---|---|---|---|---|---|
| Popularity | 0.0414 | 0.0193 | 0.0326 | 0.0000 | 0.00290 |
| GRU4Rec (CE) | 0.1255 | **0.0651** | 0.4662 | 0.0917 | **0.00079** |
| GRU4Rec + IPS | 0.1221 | 0.0641 | 0.4639 | 0.0875 | 0.00084 |
| Ours — formula | 0.1231 | 0.0638 | **0.4713** | **0.0928** | 0.00083 |
| Ours — learned | **0.1273** | 0.0649 | 0.4653 | 0.0918 | 0.00082 |

Higher is better except ΔGAP, where closer to zero is fairer.

**What the numbers actually say:**

Against uniform IPS, personalization wins fairly clearly. The formula variant beats IPS on Recall@10 (+0.8%), coverage (+1.6%) and long-tail ratio (+6.1%); the learned variant beats it on Recall@10 by 4.2%. That was the core hypothesis and it held up.

Against the plain CE baseline, the picture is more honest and more interesting. The learned variant improves Recall@10 by 1.5% but is essentially tied on NDCG@10 (−0.3%). The formula variant is *worse* on both Recall (−1.9%) and NDCG (−2.0%), and buys coverage (+1.1%) and long-tail exposure (+1.2%) with that. That's the trade-off you'd expect from a debiasing method.

**A result that didn't go our way:** plain CE has the best ΔGAP of any method, across all three user groups, and the lowest average recommended popularity. We expected debiasing to improve ΔGAP and it didn't. All the sequential methods sit in the fourth decimal, so the differences are small. Where our gains do show up consistently is coverage and long-tail ratio.

**Formula vs. learned** is a genuine trade-off, not a strict improvement. Learned takes ranking (+3.4% Recall@10, +1.8% NDCG@10 over formula); formula takes exposure (+1.3% coverage, +1.1% long-tail). Which one you'd deploy depends on whether you're optimizing engagement or catalogue health.

### Ablations

Turning off one dial at a time, **α does most of the ranking work**. That's what you'd predict for MovieLens, which has a heavy global popularity skew and comparatively mild temporal dynamics. β alone is weaker on ranking metrics but contributes to diversity, and α+β together gives the best coverage and long-tail ratio. So the two corrections aren't redundant.

**Window size** matters more than we expected. W = 7 gives the best NDCG@10 (recent trends sharpen top-rank precision) but the worst coverage, because it over-reacts to short spikes. W = 14 is the best all-round trade-off. W = 60 is worst almost everywhere. Stretch the window far enough and trending exposure just collapses back into static popularity, which defeats the point. Mean `e_trend` falls monotonically as the window grows, which is the mechanism behind that.

## Running it

```bash
pip install -r requirements.txt
```

```bash
# run the full experiment
python experiments/run_main.py --config configs/default.yaml

# run the experiment ablation 
python experiments/run_ablation.py --config configs/default.yaml

# run the sensitivity experiment
python experiments/run_sensitivity.py --config configs/default.yaml
python experiments/run_sensitivity.py --windows 7 14 30 60          # for custom windows
```

Configs live in `configs/`. The configuration that are importante:

| Key | Meaning |
|---|---|
| `loss_type` | `ce`, `ips`, `personalized`, `personalized_learned` |
| `trend_window_days` | Trending window W (7 / 14 / 30 / 60) |
| `use_alpha`, `use_beta` | Toggle each correction for ablations |
| `max_weight` | Gradient-stability clip on the debiasing weight |

Run outputs stores in `results/`.

## Limitations

**Exposure is approximated by interaction frequency.** True exposure depends on whatever recommendation policy was live when the data was logged, and that isn't observable in a public dataset. Every IPS-style method on offline data has this problem, but it's still an assumption, not a measurement.

**One dataset.** MovieLens-1M has stable preferences and mild temporal dynamics. A short-video or social feed would have far sharper trend cycles, and that's exactly the regime where β should matter most. So the fact that β underperforms here may say more about the dataset than the method.

**α needs genre metadata.** Embedding-based clustering would remove that dependency. We didn't implement it, but we have proposed this method as well.

**Offline evaluation only.** Coverage and long-tail ratio are proxies for catalogue health. Whether any of this makes recommendations *feel* better needs an online A/B test.

**Effect sizes are small and single-seed.** Most of the gaps here are in the third or fourth decimal, and we report single runs. Multiple seeds with confidence intervals would be needed before treating the ordering as settled.

## References

Key ones — full list in the report.

- Abdollahpouri et al., *The Unfairness of Popularity Bias in Recommendation*, RMSE @ RecSys 2019 — the Niche/Diverse/Blockbuster grouping and ΔGAP.
- Schnabel et al., *Recommendations as Treatments*, ICML 2016 — IPS.
- Ning et al., *Debiasing Recommendation with Personal Popularity*, WWW 2024 — closest related work; personalized but not sequential, and inference-time.
- Yang et al., *Debiasing Sequential Recommenders through DRO over System Exposure*, WSDM 2024 — sequential but uniform.
- Zhang et al., *Causal Intervention for Leveraging Popularity Bias*, SIGIR 2021 — local temporal popularity.
- Hidasi et al., *GRU4Rec*, ICLR 2016. Kang & McAuley, *SASRec*, ICDM 2018.

## Authors and contributions

Course project at IIIT Bangalore, Dept. of AI/DS and CSE.

Our group had several projects running across the semester's coursework, so we split them by ownership: each member led one project end to end, with the others contributing to problem framing, design discussions and review.

**Ritu Raj Sharma** led this project — literature review, method design, implementation, experiments, evaluation and write-up.
**M Vinay** and **Shikhar Mutta** contributed to ideation, design review and the final presentation.
