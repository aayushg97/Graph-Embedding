# Graph-Embedding

Graph embedding is an important problem in machine learning and graph theory. Given an undirected graph $G = (V, E)$ with $n$ vertices, the problem is to assign coordinates in $\mathbb{R}^m$ to each vertex $v \in V$. Typically there are desired qualities or constraints imposed on the embedding e.g. the coordinates assigned to connected nodes should be close with respect to some notion of distance. For example, the choice of Euclidean distance yields a quadratically constrained quadratic program (QCQP). Let $A \in \{0,1\}^{n\times n}$ be the symmetric adjacency of $G$, and let $D$ be the corresponding diagonal degree matrix such that $D_{ii} = \sum_{j}A_{i,j}$. The $\textit{graph Laplacian}$ is defined to be $L = D - A$.

One well-known way to do graph embedding (Laplacian Eigenmaps) is to solve the following problem:

$$\begin{equation}
\begin{aligned}
    &\min_{X \in \mathbb{R}^{n\times m}} \langle X, LX \rangle \\
    & \text{s.t. }X^\top X = I, \textbf{1}^\top X = 0
\end{aligned}
\end{equation}$$

Where the inner product $\langle A, B \rangle$ is defined to be $\text{tr}(A^\top B)$.

## Relaxation

The linear constraint may be eliminated resulting in the following relaxation

$$\begin{equation}
\begin{aligned}
    &\min_{X \in \mathbb{R}^{n\times m}} \langle X, \widetilde{L}X \rangle  \\
     & \text{s.t. } \langle X, X \rangle \leq r^2
\end{aligned}
\end{equation}$$

This can be shown as follows: Let P be the projection onto the orthogonal complement of the subspace spanned by $\textbf{1}$, i.e. $P = I − \frac{1}{n}\textbf{1}\textbf{1}^⊤$. Apply the substitution $X \leftarrow P X$. The objective can be re-written: $\langle P X, LP X \rangle = \langle X, P LP X \rangle = \langle X, \tilde{L}X \rangle$. $\tilde{L}$ is PSD. One issue with this formulation is that the relaxation admits a degenerate solution of X = 0.

## Semi-Supervised Modification

One way to condition the relaxation is to introduce additional constraints. Consider a "semi-supervised" modification of the original problem: where first $k$ vertices are "labeled" or "anchored" i.e. we have the constraints $X_i = y_i$ for $1,\ldots,k$, where $y_i \in \mathbb{R}^m$.

$$\begin{equation}
\begin{aligned}
    &\min_{X \in \mathbb{R}^{n\times m}} \langle X, \widetilde{L}X \rangle  \\
    & \text{s.t. } \langle X, X \rangle \leq r^2,\quad X_i = y_i, \ldots i = 1,\ldots, k
\end{aligned}
\end{equation}$$

