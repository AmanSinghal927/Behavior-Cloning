import tqdm
import torch
from sklearn.cluster import KMeans

class KMeansDiscretizer:
	"""
	Simplified and modified version of KMeans algorithm from sklearn.

	Code borrowed from https://github.com/notmahi/miniBET/blob/main/behavior_transformer/bet.py
	"""

	def __init__(
		self,
		num_bins: int = 100,
		kmeans_iters: int = 50,
	):
		super().__init__()
		self.n_bins = num_bins
		self.kmeans_iters = kmeans_iters

	def fit(self, input_actions: torch.Tensor) -> None: # make this custom kmeans
		self.bin_centers = KMeansDiscretizer._kmeans(
			input_actions, nbin=self.n_bins, niter=self.kmeans_iters
		)


	@classmethod
	def _kmeans(cls, x: torch.Tensor, nbin: int = 512, niter: int = 50):
		"""
		Function implementing the KMeans algorithm.

		Args:
			x: torch.Tensor: Input data - Shape: (N, D)
			nbin: int: Number of bins
			niter: int: Number of iterations
		"""

		# TODO: Implement KMeans algorithm to cluster x into nbin bins. Return the bin centers - shape (nbin, x.shape[-1])
		# if x is n-dimensional, we need x-dimensional return vector

		bin_centers = KMeans(n_clusters=nbin, max_iter=niter).fit(x).cluster_centers_
		bin_centers = torch.from_numpy(bin_centers)
		shape_bin = bin_centers.size()
		shape_x = x.size()

		return bin_centers
	
