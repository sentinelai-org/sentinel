# Sentinel Evaluation Report - ✅ PASSED

**Timestamp**: `2026-10-18T23:33:04.491919+00:00`
**Total Samples**: `8`

## Summary Metrics
| Metric | Score | Target |
| :--- | :--- | :--- |
| Mean Groundedness | **75.0%** | >= 70.0% |
| Mean Context Recall | **100.0%** | >= 70.0% |
| Mean Answer Relevance | **64.4%** | >= 60.0% |
| Gate Pass Rate | **75.0%** | >= 80.0% |
| Refusal Rate (Unanswerable) | **100.0%** | >= 80.0% |

## Sample Breakdown
| ID | Question | Groundedness | Recall | Passed |
| :--- | :--- | :--- | :--- | :--- |
| `eval_001` | What is a Pod in Kubernetes? | 100.0% | 100.0% | ✅ |
| `eval_002` | How do Kubernetes Deployments roll out update... | 100.0% | 100.0% | ✅ |
| `eval_003` | What is the difference between ClusterIP and ... | 100.0% | 100.0% | ✅ |
| `eval_004` | What are the primary control plane components... | 100.0% | 100.0% | ✅ |
| `eval_005` | What is the role of kubelet on worker nodes? | 100.0% | 100.0% | ✅ |
| `eval_006` | How does quantum teleportation entangle parti... | 100.0% | 100.0% | ✅ |
| `eval_007` | How do Services discover Pods in Kubernetes? | 0.0% | 100.0% | ❌ |
| `eval_008` | What happens when a Deployment rollback is ex... | 0.0% | 100.0% | ❌ |