# Kubernetes Architecture

A Kubernetes cluster consists of a set of worker machines, called nodes, that run containerized applications. Every cluster has at least one worker node.

The worker node(s) host the Pods that are the components of the application workload. The control plane manages the worker nodes and the Pods in the cluster. In production environments, the control plane usually runs across multiple computers and a cluster usually runs multiple nodes, providing fault-tolerance and high availability.

## Control Plane Components

- **kube-apiserver**: The API server exposes the Kubernetes API and serves as the front end for the control plane.
- **etcd**: Consistent and highly-available key value store used as Kubernetes' backing store for all cluster data.
- **kube-scheduler**: Watches for newly created Pods with no assigned node, and selects a node for them to run on.
- **kube-controller-manager**: Runs controller processes, including node controller, job controller, and endpointslice controller.
- **cloud-controller-manager**: Embeds cloud-specific control logic linking your cluster into your cloud provider's API.

## Node Components

Node components run on every node, maintaining running pods and providing the Kubernetes runtime environment:
- **kubelet**: An agent that runs on each node in the cluster, making sure containers are running in a Pod.
- **kube-proxy**: A network proxy that runs on each node in your cluster, maintaining network rules on nodes.
- **Container runtime**: The software that is responsible for running containers (such as containerd or CRI-O).
