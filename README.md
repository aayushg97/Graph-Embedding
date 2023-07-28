# Graph-Embedding

Graph embedding is an important problem in machine learning and graph theory. Given an undirected graph $G = (V, E)$ with $n$ vertices, the problem is to assign coordinates in $\mathbb{R}^m$ to each vertex $v \in V$. Typically there are desired qualities or constraints imposed on the embedding e.g. the coordinates assigned to connected nodes should be close with respect to some notion of distance. For example, the choice of Euclidean distance yields a quadratically constrained quadratic program (QCQP). Let $A \in \{0,1\}^{n\times n}$ be the symmetric adjacency of $G$, and let $D$ be the corresponding diagonal degree matrix such that $D_{ii} = \sum_{j}A_{i,j}$. The $\textit{graph Laplacian}$ is defined to be $L = D - A$.

One well known way to do graph embedding (Laplacian Eigenmaps) is to solve the following problem:

$$\begin{equation}
\begin{aligned}
    &\min_{X \in \mathbb{R}^{n\times m}} \langle X, LX \rangle \\
    & \text{s.t. }X^\top X = I, \textbf{1}^\top X = 0
\end{aligned}
\end{equation}$$

Where the inner product $\langle A, B \rangle$ is defined to be $\text{tr}(A^\top B)$.
