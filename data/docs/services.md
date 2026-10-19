# Kubernetes Services

In Kubernetes, a Service is an abstract way to expose an application running on a set of Pods as a network service.

With Kubernetes you don't need to modify your application to use an unfamiliar service discovery mechanism. Kubernetes gives Pods their own IP addresses and a single DNS name for a set of Pods, and can load-balance across them.

## Service Types

Kubernetes ServiceTypes allow you to specify what kind of Service you want:

- **ClusterIP**: Exposes the Service on an internal IP in the cluster. This type makes the Service only reachable from within the cluster. This is the default ServiceType.
- **NodePort**: Exposes the Service on each Node's IP at a static port (the NodePort). A ClusterIP Service, to which the NodePort Service routes, is automatically created. You'll be able to contact the NodePort Service, from outside the cluster, by requesting `<NodeIP>:<NodePort>`.
- **LoadBalancer**: Exposes the Service externally using a cloud provider's load balancer. NodePort and ClusterIP Services, to which the external load balancer routes, are automatically created.
- **ExternalName**: Maps the Service to the contents of the externalName field (e.g. `foo.bar.example.com`), by returning a CNAME record with its value. No proxying of any kind is set up.

## How Services Load Balance Traffic Across Pod Endpoints

Kubernetes Services do not route traffic by themselves; instead, the cluster's networking layer and the **kube-proxy** daemon implement load balancing across backend Pod endpoints.

### 1. Endpoints and EndpointSlices
When a Service is created with a selector (e.g., `app: web`), the Kubernetes EndpointSlice controller continuously queries the API Server for Pods matching those labels.
- The controller compiles the IP addresses and ports of healthy Pods into **EndpointSlice** resources.
- If a Pod fails its readiness probe or terminates, its IP address is instantly removed from the active endpoints list to prevent traffic from being directed to an unhealthy Pod.

### 2. The Role of kube-proxy
The **kube-proxy** component runs on every worker node in the Kubernetes cluster. It watches the Kubernetes API server for additions, updates, and removals of Service and EndpointSlice objects. Whenever endpoints change, `kube-proxy` dynamically configures the host node's Linux kernel networking rules to route traffic directed to the Service's virtual IP (ClusterIP) directly to the target Pod endpoint IPs.

### 3. Load Balancing Proxy Modes

`kube-proxy` operates in different proxy modes to perform this traffic distribution:

#### iptables Mode (Default)
In `iptables` mode, `kube-proxy` installs Netfilter packet filtering and NAT rules directly in the node's kernel:
- **Random Probability Load Balancing**: iptables does not use an active software balancer process. Instead, it uses the `statistic` module in iptables to select backend Pod endpoints with equal probability (e.g., probability of $1/n$ for the first endpoint, $1/(n-1)$ for the second, etc.).
- **Zero Userspace Latency**: Packets flow directly through the Linux kernel without context-switching to a userspace proxy process.
- **Failover**: If a selected Pod does not respond, connection retries depend on the client application unless the endpoint is decommissioned.

#### IPVS Mode (High Performance & Scale)
For large clusters with tens of thousands of Services, `kube-proxy` uses **IPVS (IP Virtual Server)**:
- **Kernel-level Hash Tables**: IPVS is built into the Linux kernel and uses hash tables with $O(1)$ complexity, avoiding the performance degradation that can occur with huge iptables chain rule sets ($O(n)$ traversal).
- **Load Balancing Algorithms**: IPVS supports advanced scheduling algorithms:
  - `rr` (Round-Robin)
  - `lc` (Least Connection)
  - `dh` (Destination Hashing)
  - `sh` (Source Hashing)
  - `sed` (Shortest Expected Delay)
- **Active Health Checks**: IPVS can automatically redirect traffic if a backend fails.

#### Userspace Mode (Legacy)
In this legacy mode, `kube-proxy` opens a port on the local node and forwards incoming connections to backend Pod endpoints using round-robin in user space. Because every packet must switch back and forth between kernel space and user space, this mode incurs significant latency overhead and is rarely used in modern deployments.

### 4. Session Affinity (Sticky Sessions)
By default, Kubernetes distributes requests across endpoints randomly or via round-robin. To route requests from a specific client to the same Pod each time, you can enable sticky sessions using:
```yaml
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
```
This ensures that all connections originating from the same client IP address are consistently routed to the same backend Pod until the timeout expires.

