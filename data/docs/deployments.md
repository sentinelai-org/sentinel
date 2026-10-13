# Kubernetes Deployments

A Deployment provides declarative updates for Pods and ReplicaSets in Kubernetes.

You describe a desired state in a Deployment, and the Deployment Controller changes the actual state to the desired state at a controlled rate. You can define Deployments to create new ReplicaSets, or to remove existing Deployments and adopt all their resources with new Deployments.

## Use Cases for Deployments

1. **Create a ReplicaSet to roll out Pods**: The Deployment creates a ReplicaSet in the background. Check the status of the rollout to see if it succeeds.
2. **Declare the new state of Pods**: Update the PodTemplateSpec of the Deployment. A new ReplicaSet is created and the Deployment manages moving the Pods from the old ReplicaSet to the new one at a controlled rate.
3. **Rollback to an earlier revision**: If the current state of the Deployment is not stable, you can roll back to an earlier Deployment revision.
4. **Scale up or scale down**: Scale the Deployment to facilitate more or fewer loads.
5. **Pause and resume**: Pause the rollout before applying several updates to your PodTemplateSpec and then resume it to start a new rollout.

## Rolling Update Strategy

By default, Kubernetes Deployments use the RollingUpdate strategy to ensure zero-downtime deployments. The Deployment ensures that only a certain number of Pods may be down while they are being updated (maxUnavailable), and that only a certain number of Pods may be created above the desired number of Pods (maxSurge).
