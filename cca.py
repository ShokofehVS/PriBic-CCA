"""
    PriBic-CCA: A Python library of privacy-preserving biclustering algorithm (Cheng and Church) with Function Secret
    Sharing in malicious adversary setting

    Copyright (C) 2024  Shokofeh VahidianSadegh

    This file is part of PriBic-CCA.

"""
from _base import BaseBiclusteringAlgorithm
from models import Bicluster, Biclustering
from sklearn.utils.validation import check_array
import sycret
import numpy as np
import funshade
import time
import os


class ChengChurchAlgorithm(BaseBiclusteringAlgorithm):
    """Cheng and Church's Algorithm (CCA)

    CCA searches for maximal submatrices with a Mean Squared Residue value below a pre-defined threshold.

    Reference
    ----------
    Cheng, Y., & Church, G. M. (2000). Biclustering of expression data. In Ismb (Vol. 8, No. 2000, pp. 93-103).

    Parameters
    ----------
    num_biclusters : int, default: 100
        Number of biclusters to be found.

    msr_threshold : float, default: 300 or 1200
        Maximum mean squared residue accepted (delta parameter in the original paper).

    multiple_node_deletion_threshold : float, default: 1.2
        Scaling factor to remove multiple rows or columns (alpha parameter in the original paper).

    data_min_cols : int, default: 100
        Minimum number of dataset columns required to perform multiple column deletion.
    """

    def __init__(self, high, num_biclusters, msr_threshold, multiple_node_deletion_threshold, data_min_cols):
        self.highest_range = high
        self.num_biclusters = num_biclusters
        self.msr_threshold = msr_threshold
        self.multiple_node_deletion_threshold = multiple_node_deletion_threshold
        self.data_min_cols = data_min_cols

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """

        # Create parties
        class party:
            def __init__(self, d: int):
                self.d = d

        P_0 = party(0)
        P_1 = party(1)
        P_2 = party(2)

        # Check input data
        data = check_array(data, dtype=int, copy=True)

        # Helper vectors
        biclusters = []
        t_shareMSR, t_shareEval, t_sdel, t_muldel, t_add, t_size = [], [], [], [], [], []

        # For number of biclusters do the steps:
        for i in range(self.num_biclusters):
            # Shape of data and min/ max of that
            num_rows, num_cols = data.shape
            min_value          = np.min(data)
            max_value          = np.max(data)

            # Generate secret shares of the input data
            rng  = np.random.default_rng(seed=42)
            in_0 = rng.integers(0, self.highest_range, size=(num_rows, num_cols), dtype="int64")
            in_1 = rng.integers(0, self.highest_range, size=(num_rows, num_cols), dtype="int64")
            in_2 = data - in_0 - in_1

            # Assign replicated secret shares to parties
            P_0.aij_0 = np.copy(in_0)
            P_0.aij_1 = np.copy(in_1)

            P_1.aij_1 = np.copy(in_1)
            P_1.aij_2 = np.copy(in_2)

            P_2.aij_2 = np.copy(in_2)
            P_2.aij_0 = np.copy(in_0)

            # Shape of inputs for both parties
            num_row_0, num_col_0 = in_0.shape
            num_row_1, num_col_1 = in_1.shape
            num_row_2, num_col_2 = in_2.shape

            # Steps including single, multiple deletion/ addition
            # P_0.bij_0, P_1.bij_1, P_2.bij_2, len_row  = self._multiple_node_deletion(P_0, P_1, P_2, in_0, in_1, in_2,
            #                                                                          self.msr_threshold)

            # P_0.cij_0, P_1.cij_1, len_row, len_col  = (
            #     self._single_node_deletion(P_0, P_1, P_2, P_0.bij_0, P_1.bij_1, P_2.bij_2, len_row, self.msr_threshold))

            P_0.cij_0, P_1.cij_1, len_row, len_col = (
                self._single_node_deletion(P_0, P_1, P_2, in_0, in_1, in_2, num_row_0, self.msr_threshold))
            #
            # P_0.dij_0, P_1.dij_1, len_row, len_col = (
            #     self._node_addition(P_0.cij_0, P_1.cij_1, in_0, in_1, len_row, len_col, t_shareMSR, t_shareEval, t_add,t_size))


            # Output shares then be reconstructed as the final matrix
            new_data = P_0.dij_0 + P_1.dij_1

            # Rows and columns indexes without zeros
            rows_without_zeros   = ~np.any(new_data == 0, axis=1)
            cols_without_zeros   = ~np.any(new_data == 0, axis=0)
            rows_indexes         = np.where(rows_without_zeros)[0]
            cols_indexes         = np.where(cols_without_zeros)[0]

            if len_row == 0 or len_col == 0:
                break

            # Masking matrix values
            if i < self.num_biclusters - 1:
                bicluster_shape = (len_row, len_col)
                data            = np.random.uniform(low=min_value, high=max_value, size=bicluster_shape)

            biclusters.append(Bicluster(rows_indexes, cols_indexes))


        return Biclustering(biclusters)


    def _single_node_deletion(self, P_0, P_1, P_2, in_0, in_1, in_2, len_row, msr_thr):
        """Performs the single row/column deletion step (this is a direct implementation of the Algorithm 1 described
           in the original paper)"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape
        num_row_1, num_col_1 = in_1.shape
        num_row_2, num_col_2 = in_2.shape

        len_col = num_col_0

        # Calculate the scores by having inputs including secret shares of matrix, and length of rows, columns
        msr_0, row_msr_0, col_msr_0, msr_1, row_msr_1, col_msr_1, msr_2, row_msr_2, col_msr_2 = \
            (self._scores_after_steps(P_0, P_1, P_2, in_0, in_1, in_2, len_row, num_col_0))

        # STOP function -- Check whether the MSR is below or equal to threshold
        stop_itr_0 = 0 - msr_0
        stop_itr_1 = 0 - msr_1
        stop_itr_2 = msr_thr - msr_2

        stop0 = self.fss_evaluation(stop_itr_0, stop_itr_1, 1, 0)
        stop1 = self.fss_evaluation(stop_itr_1, stop_itr_2, 1, 0)
        stop2 = self.fss_evaluation(stop_itr_2, stop_itr_0, 1, 0)

        stop  = (stop0 & stop1) | (stop1 & stop2) | (stop2 & stop0)

        if stop:
            # No node has been removed
            return in_0, in_1, in_2, len_row, num_col_0

        else:
            while not stop:
                # Find the argmax of nodes whichever having the largest scores
                row_max_msr0 = self._amx(row_msr_0, row_msr_1)
                row_max_msr1 = self._amx(row_msr_1, row_msr_2)
                row_max_msr2 = self._amx(row_msr_2, row_msr_0)
                row_max = row_max_msr0 if (row_max_msr0 == row_max_msr1 or row_max_msr0 == row_max_msr2) else row_max_msr1

                col_max_msr0 = self._amx(col_msr_0, col_msr_1)
                col_max_msr1 = self._amx(col_msr_1, col_msr_2)
                col_max_msr2 = self._amx(col_msr_2, col_msr_0)
                col_max = col_max_msr0 if (col_max_msr0 == col_max_msr1 or col_max_msr0 == col_max_msr2) else col_max_msr1

                # Check score of row/ column with maximum values to remove that particular node (focus on *columns*)
                eval_col_0 = row_msr_0[row_max] - col_msr_0[col_max]
                eval_col_1 = row_msr_1[row_max] - col_msr_1[col_max]
                eval_col_2 = row_msr_2[row_max] - col_msr_2[col_max]

                sdel_00, sdel_01 = self.fss_evaluation(eval_col_0, eval_col_1, 1, 1)
                sdel_10, sdel_11 = self.fss_evaluation(eval_col_1, eval_col_2, 1, 1)
                sdel_20, sdel_21 = self.fss_evaluation(eval_col_2, eval_col_0, 1, 1)

                sdel_0   = sdel_00 + sdel_01;        sdel_1 = sdel_10 + sdel_11;             sdel_2 = sdel_20 + sdel_21
                fss_cols = ((sdel_0 & sdel_1) | (sdel_1 & sdel_2) | (sdel_2 & sdel_0))

                # Remove single column based on the eval_col result
                # So that transpose secret shared input matrices before removing column by multiplying fss_cols with them
                transposed_in_0 = in_0.T;          transposed_in_1 = in_1.T;               transposed_in_2 = in_2.T
                transposed_in_0[col_max_msr0] *= fss_cols
                transposed_in_1[col_max_msr1] *= fss_cols
                transposed_in_1[col_max_msr2] *= fss_cols

                # Update the length of the columns
                len_col = len_col + (fss_cols - 1)

                # Return the transposed matrices to normal
                in_0 = transposed_in_0.T;          in_1 = transposed_in_1.T;             in_2 = transposed_in_2.T

                # Check score of row/ column with maximum values to remove that particular node (focus on *rows*)
                eval_row_0 = col_msr_0[col_max] - row_msr_0[row_max]
                eval_row_1 = col_msr_1[col_max] - row_msr_1[row_max]
                eval_row_2 = col_msr_2[col_max] - row_msr_2[row_max]

                rdel_00, rdel_01 = self.fss_evaluation(eval_row_0, eval_row_1, 1, 1)
                rdel_10, rdel_11 = self.fss_evaluation(eval_row_1, eval_row_2, 1, 1)
                rdel_20, rdel_21 = self.fss_evaluation(eval_row_2, eval_row_0, 1, 1)

                rdel_0   = rdel_00 + rdel_01;         rdel_1 = rdel_10 + rdel_11;            rdel_2 = rdel_20 + rdel_21
                fss_rows = ((rdel_0 & rdel_1) | (rdel_1 & rdel_2) | (rdel_2 & rdel_0))

                # Remove single row based on the eval_row result by multiplying fss_rows with input shares
                in_0[row_max_msr0] *= fss_rows;     in_1[row_max_msr1] *= fss_rows;       in_2[row_max_msr2] *= fss_rows

                # Update the length of the rows
                len_row  = len_row + (fss_rows - 1)

                # Recalculate the scores by having inputs including secret shares of matrix, and length of rows, columns
                msr_0, row_msr_0, col_msr_0, msr_1, row_msr_1, col_msr_1, msr_2, row_msr_2, col_msr_2 = \
                    (self._scores_after_steps(P_0, P_1, P_2, in_0, in_1, in_2, len_row, num_col_0))

                # Recheck the stop function
                stop_itr_0 = 0 - msr_0
                stop_itr_1 = 0 - msr_1
                stop_itr_2 = msr_thr - msr_2
                stop0 = self.fss_evaluation(stop_itr_0, stop_itr_1, 1, 0)
                stop1 = self.fss_evaluation(stop_itr_1, stop_itr_2, 1, 0)
                stop2 = self.fss_evaluation(stop_itr_2, stop_itr_0, 1, 0)

                stop = (stop0 & stop1) | (stop1 & stop2) | (stop2 & stop0)


        return in_0, in_1, len_row, num_col_0


    def _multiple_node_deletion(self, P_0, P_1, P_2, in_0, in_1, in_2, msr_thr):
        """Performs the multiple row/column deletion step (this is a direct implementation of the Algorithm 2 described
           in the original paper)"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape

        # Initialization -- row size
        total_len_row = num_row_0

        # MSRs computation when NO nodes are removed (having exact rows/ columns length)
        msr_0, row_msr_0, col_msr_0, msr_1, row_msr_1, col_msr_1, msr_2, row_msr_2, col_msr_2 = (
            self._scores_before_steps(P_0, P_1, P_2, in_0, in_1, in_2))

        # STOP function -- Check whether the MSR is below or equal to threshold
        stop_itr_0 = 0 - msr_0
        stop_itr_1 = 0 - msr_1
        stop_itr_2 = msr_thr - msr_2

        stop0      = self.fss_evaluation(stop_itr_0, stop_itr_1, 1, 0)
        stop1      = self.fss_evaluation(stop_itr_1, stop_itr_2, 1, 0)
        stop2      = self.fss_evaluation(stop_itr_2, stop_itr_0, 1, 0)

        stop       = (stop0 & stop1) | (stop1 & stop2) | (stop2 & stop0)
        no_itr     = 0
        # stop_initr = np.log(num_row_0)+np.log(num_col_0)
        stop_initr = 1

        if stop:
            # No nodes have been removed so return length of rows without change
            return in_0, in_1, in_2, num_row_0

        else:
            while no_itr < stop_initr:
                # Store previous values of matrices for equality check
                cp_in_0 = np.copy(in_0);    cp_in_1 = np.copy(in_1);    cp_in_2 = np.copy(in_2)

                # FSS IC gate to check which rows should be removed
                r2remove_con_0 = self.multiple_node_deletion_threshold * msr_0 - row_msr_0
                r2remove_con_1 = self.multiple_node_deletion_threshold * msr_1 - row_msr_1
                r2remove_con_2 = self.multiple_node_deletion_threshold * msr_2 - row_msr_2
                r2remove_con = r2remove_con_2+r2remove_con_1+r2remove_con_0

                fss_rs_rows_00, fss_rs_rows_01 = self.fss_evaluation(r2remove_con_0, r2remove_con_1, None, 0)
                fss_rs_rows_10, fss_rs_rows_11 = self.fss_evaluation(r2remove_con_1, r2remove_con_2, None, 0)
                fss_rs_rows_20, fss_rs_rows_21 = self.fss_evaluation(r2remove_con_2, r2remove_con_0, None, 0)

                # Remove the rows based on the result of fss (inputs * fss)
                fss_rs_rows0 = fss_rs_rows_00 + fss_rs_rows_01
                fss_rs_rows1 = fss_rs_rows_10 + fss_rs_rows_11
                fss_rs_rows2 = fss_rs_rows_20 + fss_rs_rows_21

                fss_rs_rows     = ((fss_rs_rows0 & fss_rs_rows1) | (fss_rs_rows1 & fss_rs_rows2) |
                                   (fss_rs_rows2 & fss_rs_rows0))
                fss_rs_rows_rot = fss_rs_rows[:, np.newaxis]

                in_0 *=  fss_rs_rows_rot;       in_1 *= fss_rs_rows_rot;      in_2 *= fss_rs_rows_rot

                # Update the length of the rows based on the sum of fss
                total_len_row = sum(fss_rs_rows)

                # Check whether columns are above 100 then apply node deletion on them
                if num_col_0 >= self.data_min_cols:
                    pass

                # Recalculate the scores (the columns by default have not been removed) to be used in stop function below
                msr_0, row_msr_0, col_msr_0, msr_1, row_msr_1, col_msr_1, msr_2, row_msr_2, col_msr_2 =  \
                    (self._scores_after_steps(P_0, P_1, P_2, in_0, in_1, in_2, total_len_row, num_col_0))

        #         BRING BACK THE CONDITIONS

        return in_0, in_1, in_2, total_len_row


    def _node_addition(self, bij_0, bij_1, in_0, in_1, total_len_row, len_col):
        """Performs the row/column addition step (this is a direct implementation of the Algorithm 3 described in
           the original paper)"""
        # Secret shared inputs' shapes
        t_add_0 = time.perf_counter()

        num_row_0, num_col_0 = bij_0.shape
        num_row_1, num_col_1 = bij_1.shape

        # Calculate score for whole matrix and that of columns
        t_msr_0 = time.perf_counter()

        msr_0, msr_1, _, _, _, _ = (self._scores_after_steps(bij_0, bij_1, total_len_row, len_col))
        col_msr_0, col_msr_1     = (self._scores_column_addition(bij_0, bij_1, total_len_row, len_col))

        t_msr_1 = time.perf_counter()
        t_shareMSR.append(t_msr_1 - t_msr_0)

        # FSS IC gate to check which columns should be added
        t_eval_0 = time.perf_counter()

        r2add_con_0 = msr_0 - col_msr_0
        r2add_con_1 = msr_1 - col_msr_1
        fss_rs_col_add_0, fss_rs_col_add_1 = self.fss_evaluation_without_len(r2add_con_0, r2add_con_1, t_size)

        # Then follow similar steps in deletion; except return those have been removed
        # Add the columns based on the result of evaluation 1 => nothing, 0 => add columns
        nc2add = self._equality_check_2(fss_rs_col_add_0, fss_rs_col_add_1, 0, 0, num_col_0,t_size)

        # Transpose secret shared input matrices before adding column (original, and result of previous step)
        transposed_bin_0 = bij_0.T
        transposed_bin_1 = bij_1.T

        transposed_in_0  = in_0.T
        transposed_in_1  = in_1.T

        # Because some columns might be zero now, let's ignore them in addition
        # Finds those zeros columns and try to add them with original matrix values
        for idxc in range(num_col_0):
            scadd = self._equality_check_2(transposed_bin_0[idxc], transposed_bin_1[idxc], 0, 0, num_row_0,t_size)
            i = nc2add[idxc]
            if i == 0 and scadd.all() == 1:
                transposed_bin_0[idxc] = transposed_in_0[idxc]
                transposed_bin_1[idxc] = transposed_in_1[idxc]
                len_col += 1
            else:
                pass

        # Return the transposed matrices to normal
        in_0  = transposed_in_0.T
        in_1  = transposed_in_1.T

        bij_0 = transposed_bin_0.T
        bij_1 = transposed_bin_1.T

        t_eval_1 = time.perf_counter()
        t_shareEval.append(t_eval_1 - t_eval_0)

        # Calculate score for whole matrix and that of rows
        t_msr_0 = time.perf_counter()

        msr_0, msr_1, _, _, _, _ =  (self._scores_after_steps(bij_0, bij_1, total_len_row, len_col,t_size))
        row_msr_0, row_msr_1     =  (self._scores_row_addition(bij_0, bij_1, total_len_row, len_col,t_size))

        t_msr_1 = time.perf_counter()
        t_shareMSR.append(t_msr_1 - t_msr_0)

        # FSS IC gate to check which rows should be added
        t_eval_0 = time.perf_counter()

        r2add_con_0 = msr_0 - row_msr_0
        r2add_con_1 = msr_1 - row_msr_1
        fss_rs_rows_add_0, fss_rs_rows_add_1 = self.fss_evaluation_without_len(r2add_con_0, r2add_con_1,t_size)

        # Then follow similar steps in deletion
        # Add the rows based on the result of evaluation 1 => nothing, 0 => add rows
        nr2add = self._equality_check_2(fss_rs_rows_add_0, fss_rs_rows_add_1, 0, 0, num_row_0,t_size)

        # Because some rows might be zero now, let's ignore them in addition
        # Finds those zeros rows and try to add them with original matrix values
        for idxr in range(num_row_0):
            sradd = self._equality_check_2(bij_0[idxr], bij_1[idxr], 0, 0, num_col_0,t_size)
            i = nr2add[idxr]
            if i == 0 and sradd.all() == 1:
                bij_0[idxr] = in_0[idxr]
                bij_1[idxr] = in_1[idxr]
                total_len_row += 1
            else:
                pass

        t_eval_1 = time.perf_counter()
        t_shareEval.append(t_eval_1 - t_eval_0)

        t_add_1 = time.perf_counter()
        t_add.append(t_add_1 - t_add_0)

        return bij_0, bij_1, total_len_row, len_col

    def _amx(self, in_0, in_1):
        """Calculate Argmax of scores of the rows, of the columns."""
        # Initial values
        m = len(in_0)
        argmx_con_0, argmx_con_1, delta_j  = [], [], []

        # Argmax according to AriaNN algorithm 6
        for j in range(m):
            for i in range(m):
                if i != j:
                    argmx_con_0.append(in_0[j] - in_0[i])
                    argmx_con_1.append(in_1[j] - in_1[i])
                else:
                    pass
            node_max_0, node_max_1 = self.fss_evaluation(np.array(argmx_con_0), np.array(argmx_con_1),None, 0)
            s_j_0 = np.sum(node_max_0);                                 s_j_1 = np.sum(node_max_1)
            delta_j.append(self._equality_check_2(s_j_0, s_j_1, m-1, 0, 1))
            argmx_con_0, argmx_con_1 = [], []
            if delta_j[j] == 1:
                arg_max_res = j
                return arg_max_res


    def _scores_before_steps(self, P0, P1, P2, in_0, in_1, in_2):
        """Calculate scores of the rows, of the columns and of the full data matrix before any steps"""
        # Mean values for whole data, rows and columns
        mu_ij_0, mu_r_0, mu_c_0, mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2 = self.secMean(P0, P1, P2,
                                                                                                 in_0, in_1, in_2)

        # Residue for the input matrix
        r_ij_0, r_ij_1, r_ij_2 = self.secResidue(P0, P1, P2, in_0, in_1, in_2, mu_ij_0, mu_r_0, mu_c_0,
                                                          mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2)

        # Continue doing squaring by joint multiplication
        r2_ij_0, r2_ij_1, r2_ij_2 = self.secSquaring(P0, P1, P2, r_ij_0, r_ij_1, r_ij_2)

        # MSRs for whole data, rows and columns
        h_ij_0, h_r_0, h_c_0, h_ij_1, h_r_1, h_c_1, h_ij_2, h_r_2, h_c_2 = self.secMean(P0, P1, P2,
                                                                                        r2_ij_0, r2_ij_1, r2_ij_2)

        return h_ij_0, h_r_0, h_c_0, h_ij_1, h_r_1, h_c_1, h_ij_2, h_r_2, h_c_2


    def _scores_after_steps(self, P0, P1, P2, in_0, in_1, in_2, len_row, len_col):
        """Calculate scores of the rows, of the columns and of the full data matrix after node changes"""
        # Mean values for whole data, rows and columns
        mu_ij_0, mu_r_0, mu_c_0, mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2 = self.secSum(P0, P1, P2,
                                                                                                 in_0, in_1, in_2,
                                                                                                len_row, len_col)

        # Residue for the input matrix
        r_ij_0, r_ij_1, r_ij_2 = self.secResidue(P0, P1, P2, in_0, in_1, in_2, mu_ij_0, mu_r_0, mu_c_0,
                                                          mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2)

        # Continue doing squaring by joint multiplication
        r2_ij_0, r2_ij_1, r2_ij_2 = self.secSquaring(P0, P1, P2, r_ij_0, r_ij_1, r_ij_2)

        # MSRs for whole data, rows and columns
        h_ij_0, h_r_0, h_c_0, h_ij_1, h_r_1, h_c_1, h_ij_2, h_r_2, h_c_2 = self.secMean(P0, P1, P2,
                                                                                        r2_ij_0, r2_ij_1, r2_ij_2)

        return h_ij_0, h_r_0, h_c_0, h_ij_1, h_r_1, h_c_1, h_ij_2, h_r_2, h_c_2


    def _scores_column_addition(self, P0, P1, P2, in_0, in_1, in_2, len_row, len_col):
        """Calculate scores of the columns for node addition step"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape

        # Mean values for whole data, rows and columns
        mu_ij_0, mu_r_0, mu_c_0, mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2 = self.secSumCol(P0, P1, P2,
                                                                                                in_0, in_1, in_2,
                                                                                                len_row, num_row_0,
                                                                                                   len_col)

        # Residue for the input matrix
        r_ij_0, r_ij_1, r_ij_2 = self.secResidue(P0, P1, P2, in_0, in_1, in_2, mu_ij_0, mu_r_0, mu_c_0,
                                                 mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2)

        # Continue doing squaring by joint multiplication
        r2_ij_0, r2_ij_1, r2_ij_2 = self.secSquaring(P0, P1, P2, r_ij_0, r_ij_1, r_ij_2)

        # MSRs for columns only
        h_c_0, h_c_1, h_c_2 = self.secMeanCol(P0, P1, P2, r2_ij_0, r2_ij_1, r2_ij_2)

        return h_c_0, h_c_1, h_c_2


    def _scores_row_addition(self, P0, P1, P2, in_0, in_1, in_2, len_row, len_col):
        """Calculate scores of the rows for node addition"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape

        # Mean values for whole data, rows and columns
        mu_ij_0, mu_r_0, mu_c_0, mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2 = self.secSumRow(P0, P1, P2,
                                                                                                in_0, in_1, in_2,
                                                                                                len_row,
                                                                                                len_col, num_col_0)

        # Residue for the input matrix
        r_ij_0, r_ij_1, r_ij_2 = self.secResidue(P0, P1, P2, in_0, in_1, in_2, mu_ij_0, mu_r_0, mu_c_0,
                                                 mu_ij_1, mu_r_1, mu_c_1, mu_ij_2, mu_r_2, mu_c_2)

        # Continue doing squaring by joint multiplication
        r2_ij_0, r2_ij_1, r2_ij_2 = self.secSquaring(P0, P1, P2, r_ij_0, r_ij_1, r_ij_2)

        # MSRs for rows only
        h_r_0, h_r_1, h_r_2 = self.secMeanRow(P0, P1, P2, r2_ij_0, r2_ij_1, r2_ij_2)

        return  h_r_0, h_r_1, h_r_2


    def _equality_check(self, in_0, in_1, cp_in_0, cp_in_1):
        """Determine equality of matrix before and after node deletion; usage in stop function of multiple deletion"""
        # Determine the number of secret shared elements for keys
        n_row, n_cols = in_0.shape
        n_element     = n_row * n_cols

        # An instance of DPF gate for equality check with 6 threads
        eq = sycret.EqFactory(n_threads=6)

        # Generation of DPF keys
        keys_a, keys_b = eq.keygen(n_element)

        # Alpha based on generated keys
        alpha = eq.alpha(keys_a, keys_b)

        # Secret share the Alpha
        rng = np.random.default_rng(seed=42)
        e_rin_0 = rng.integers(1, self.highest_range, size=n_element, dtype="int64")
        e_rin_1 = alpha - e_rin_0

        # Input shares for DPF gate
        dpf_in0 = in_0 - cp_in_0
        dpf_in1 = in_1 - cp_in_1

        # Convert to flatten vectors
        dpf_in0 = dpf_in0.flatten()
        dpf_in1 = dpf_in1.flatten()

        # Add the mask to secret shares before reconstruction
        mdpf_in0 = dpf_in0 + e_rin_0
        mdpf_in1 = dpf_in1 + e_rin_1

        # Now exchange the masked input to DPF FSS gate
        f_out = mdpf_in0 + mdpf_in1

        # Apply DPF for equality check
        r_a, r_b = (
            eq.eval(0, f_out, keys_a),
            eq.eval(1, f_out, keys_b),
        )
        r_eq = (r_a + r_b) % (2 ** (eq.N * 8))

        # Check whether all nodes are the same or there are any changes in the matrices
        if np.sum(r_eq) == n_element:
            stop = True
        else:
            stop = False

        return stop


    # def _equality_check_2(self, in_0, in_1, cp_in_0, cp_in_1, n_element,t_size):
    #     """Determine equality of vectors; usage in deletion steps"""
    #     # An instance of DPF gate for equality check with 6 threads
    #     eq = sycret.EqFactory(n_threads=6)
    #
    #     # Generation of DPF keys
    #     keys_a, keys_b = eq.keygen(n_element)
    #
    #     # Alpha based on generated keys
    #     alpha = eq.alpha(keys_a, keys_b)
    #
    #     # Secret share the Alpha
    #     rng = np.random.default_rng(seed=42)
    #     e_rin_0 = rng.integers(1, self.highest_range, size=n_element, dtype="int64")
    #     e_rin_1 = alpha - e_rin_0
    #
    #     # Input shares for DPF gate
    #     dpf_in0 = in_0 - cp_in_0
    #     dpf_in1 = in_1 - cp_in_1
    #
    #     # Add the mask to secret shares before reconstruction
    #     mdpf_in0 = dpf_in0 + e_rin_0
    #     mdpf_in1 = dpf_in1 + e_rin_1
    #
    #     # Now exchange the masked input to DPF FSS gate
    #     f_out = mdpf_in0 + mdpf_in1
    #
    #     with open('result_size.txt', 'w') as saveFile:
    #         saveFile.write(str(mdpf_in0) + "\n")
    #         saveFile.write(str(mdpf_in1) + "\n")
    #     t_size.append(os.path.getsize("result_size.txt"))
    #
    #     # Apply DPF for equality check
    #     r_a, r_b = (
    #         eq.eval(0, f_out, keys_a),
    #         eq.eval(1, f_out, keys_b),
    #     )
    #     r_eq = (r_a + r_b) % (2 ** (eq.N * 8))
    #
    #     return r_eq
    #
    def _equality_check_2(self, in_0, in_1, cp_in_0, cp_in_1, n_element):
        """Determine equality of vectors; usage in deletion steps"""
        # An instance of DPF gate for equality check with 6 threads
        eq = sycret.EqFactory(n_threads=6)

        # Generation of DPF keys
        keys_a, keys_b = eq.keygen(n_element)

        # Alpha based on generated keys
        alpha = eq.alpha(keys_a, keys_b)

        # Secret share the Alpha
        rng = np.random.default_rng(seed=42)
        e_rin_0 = rng.integers(1, self.highest_range, size=n_element, dtype="int64")
        e_rin_1 = alpha - e_rin_0

        # Input shares for DPF gate
        dpf_in0 = in_0 - cp_in_0
        dpf_in1 = in_1 - cp_in_1

        # Add the mask to secret shares before reconstruction
        mdpf_in0 = dpf_in0 + e_rin_0
        mdpf_in1 = dpf_in1 + e_rin_1

        # Now exchange the masked input to DPF FSS gate
        f_out = mdpf_in0 + mdpf_in1

        # Apply DPF for equality check
        r_a, r_b = (
            eq.eval(0, f_out, keys_a),
            eq.eval(1, f_out, keys_b),
        )
        r_eq = (r_a + r_b) % (2 ** (eq.N * 8))

        return r_eq


    def secSum(self, P0, P1, P2, in_0, in_1, in_2, len_row, len_col):
        """Secured sum based on RSS consisting of local linear and non-linear functions."""
        # Get sum values for whole matrix, row-wise and column-wise then divide it locally to size of matrix
        P0.mu_ij_0 = (np.sum(in_0) / (len_row * len_col)).astype(int)
        P0.mu_r_0  = (np.sum(in_0, axis=1) / len_col).astype(int)
        P0.mu_c_0  = (np.sum(in_0, axis=0) / len_row).astype(int)

        P1.mu_ij_1 = (np.sum(in_1) / (len_row * len_col)).astype(int)
        P1.mu_r_1  = (np.sum(in_1, axis=1) / len_col).astype(int)
        P1.mu_c_1  = (np.sum(in_1, axis=0) / len_row).astype(int)

        P2.mu_ij_2 = (np.sum(in_2) / (len_row * len_col)).astype(int)
        P2.mu_r_2  = (np.sum(in_2, axis=1) / len_col).astype(int)
        P2.mu_c_2  = (np.sum(in_2, axis=0) / len_row).astype(int)

        # RSS shares for each parties
        P0.mu_ij_1 = np.copy(P1.mu_ij_1)
        P0.mu_r_1  = np.copy(P1.mu_r_1)
        P0.mu_c_1  = np.copy(P1.mu_c_1)

        P1.mu_ij_2 = np.copy(P2.mu_ij_2)
        P1.mu_r_2  = np.copy(P2.mu_r_2)
        P1.mu_c_2  = np.copy(P2.mu_c_2)

        P2.mu_ij_0 = np.copy(P0.mu_ij_0)
        P2.mu_r_0  = np.copy(P0.mu_r_0)
        P2.mu_c_0  = np.copy(P0.mu_c_0)

        return P0.mu_ij_0, P0.mu_r_0, P0.mu_c_0, P1.mu_ij_1, P1.mu_r_1, P1.mu_c_1, P2.mu_ij_2, P2.mu_r_2, P2.mu_c_2


    def secSumCol(self, P0, P1, P2, in_0, in_1, in_2, len_row, num_row, len_col):
        """Secured sum based on RSS consisting of local linear and non-linear functions."""
        # Get sum values for whole matrix, row-wise and column-wise then divide it locally to size of matrix
        P0.mu_ij_0 = (np.sum(in_0) / (len_row * len_col)).astype(int)
        P0.mu_r_0 = (np.sum(in_0, axis=1) / len_col).astype(int)
        P0.mu_c_0 = (np.sum(in_0, axis=0) / num_row).astype(int)

        P1.mu_ij_1 = (np.sum(in_1) / (len_row * len_col)).astype(int)
        P1.mu_r_1 = (np.sum(in_1, axis=1) / len_col).astype(int)
        P1.mu_c_1 = (np.sum(in_1, axis=0) / num_row).astype(int)

        P2.mu_ij_2 = (np.sum(in_2) / (len_row * len_col)).astype(int)
        P2.mu_r_2 = (np.sum(in_2, axis=1) / len_col).astype(int)
        P2.mu_c_2 = (np.sum(in_2, axis=0) / num_row).astype(int)

        # RSS shares for each parties
        P0.mu_ij_1 = np.copy(P1.mu_ij_1)
        P0.mu_r_1 = np.copy(P1.mu_r_1)
        P0.mu_c_1 = np.copy(P1.mu_c_1)

        P1.mu_ij_2 = np.copy(P2.mu_ij_2)
        P1.mu_r_2 = np.copy(P2.mu_r_2)
        P1.mu_c_2 = np.copy(P2.mu_c_2)

        P2.mu_ij_0 = np.copy(P0.mu_ij_0)
        P2.mu_r_0 = np.copy(P0.mu_r_0)
        P2.mu_c_0 = np.copy(P0.mu_c_0)

        return P0.mu_ij_0, P0.mu_r_0, P0.mu_c_0, P1.mu_ij_1, P1.mu_r_1, P1.mu_c_1, P2.mu_ij_2, P2.mu_r_2, P2.mu_c_2


    def secSumRow(self, P0, P1, P2, in_0, in_1, in_2, len_row, len_col, num_col):
        """Secured sum based on RSS consisting of local linear and non-linear functions."""
        # Get sum values for whole matrix, row-wise and column-wise then divide it locally to size of matrix
        P0.mu_ij_0 = (np.sum(in_0) / (len_row * len_col)).astype(int)
        P0.mu_r_0 = (np.sum(in_0, axis=1) / num_col).astype(int)
        P0.mu_c_0 = (np.sum(in_0, axis=0) / len_row).astype(int)

        P1.mu_ij_1 = (np.sum(in_1) / (len_row * len_col)).astype(int)
        P1.mu_r_1 = (np.sum(in_1, axis=1) / num_col).astype(int)
        P1.mu_c_1 = (np.sum(in_1, axis=0) / len_row).astype(int)

        P2.mu_ij_2 = (np.sum(in_2) / (len_row * len_col)).astype(int)
        P2.mu_r_2 = (np.sum(in_2, axis=1) / num_col).astype(int)
        P2.mu_c_2 = (np.sum(in_2, axis=0) / len_row).astype(int)

        # RSS shares for each parties
        P0.mu_ij_1 = np.copy(P1.mu_ij_1)
        P0.mu_r_1 = np.copy(P1.mu_r_1)
        P0.mu_c_1 = np.copy(P1.mu_c_1)

        P1.mu_ij_2 = np.copy(P2.mu_ij_2)
        P1.mu_r_2 = np.copy(P2.mu_r_2)
        P1.mu_c_2 = np.copy(P2.mu_c_2)

        P2.mu_ij_0 = np.copy(P0.mu_ij_0)
        P2.mu_r_0 = np.copy(P0.mu_r_0)
        P2.mu_c_0 = np.copy(P0.mu_c_0)

        return P0.mu_ij_0, P0.mu_r_0, P0.mu_c_0, P1.mu_ij_1, P1.mu_r_1, P1.mu_c_1, P2.mu_ij_2, P2.mu_r_2, P2.mu_c_2


    def secMean(self, P0, P1, P2, in_0, in_1, in_2):
        """Secured mean based on RSS consisting of local linear and non-linear functions."""
        # Get mean values for whole matrix, row-wise and column-wise
        P0.mu_ij_0 = np.mean(in_0).astype(int)
        P0.mu_r_0  = np.mean(in_0, axis=1).astype(int)
        P0.mu_c_0  = np.mean(in_0, axis=0).astype(int)

        P1.mu_ij_1 = np.mean(in_1).astype(int)
        P1.mu_r_1  = np.mean(in_1, axis=1).astype(int)
        P1.mu_c_1  = np.mean(in_1, axis=0).astype(int)

        P2.mu_ij_2 = np.mean(in_2).astype(int)
        P2.mu_r_2  = np.mean(in_2, axis=1).astype(int)
        P2.mu_c_2  = np.mean(in_2, axis=0).astype(int)

        # RSS shares for each parties
        P0.mu_ij_1 = np.copy(P1.mu_ij_1)
        P0.mu_r_1  = np.copy(P1.mu_r_1)
        P0.mu_c_1  = np.copy(P1.mu_c_1)

        P1.mu_ij_2 = np.copy(P2.mu_ij_2)
        P1.mu_r_2  = np.copy(P2.mu_r_2)
        P1.mu_c_2  = np.copy(P2.mu_c_2)

        P2.mu_ij_0 = np.copy(P0.mu_ij_0)
        P2.mu_r_0  = np.copy(P0.mu_r_0)
        P2.mu_c_0  = np.copy(P0.mu_c_0)

        return P0.mu_ij_0, P0.mu_r_0, P0.mu_c_0, P1.mu_ij_1, P1.mu_r_1, P1.mu_c_1, P2.mu_ij_2, P2.mu_r_2, P2.mu_c_2


    def secMeanCol(self, P0, P1, P2, in_0, in_1, in_2):
        """Secured mean based on RSS consisting of local linear and non-linear functions."""
        # Get mean values column-wise
        P0.mu_c_0  = np.mean(in_0, axis=0).astype(int)

        P1.mu_c_1  = np.mean(in_1, axis=0).astype(int)

        P2.mu_c_2  = np.mean(in_2, axis=0).astype(int)

        # RSS shares for each parties
        P0.mu_c_1  = np.copy(P1.mu_c_1)

        P1.mu_c_2  = np.copy(P2.mu_c_2)

        P2.mu_c_0  = np.copy(P0.mu_c_0)

        return P0.mu_c_0, P1.mu_c_1, P2.mu_c_2


    def secMeanRow(self, P0, P1, P2, in_0, in_1, in_2):
        """Secured mean based on RSS consisting of local linear and non-linear functions."""
        # Get mean values row-wise
        P0.mu_r_0 = np.mean(in_0, axis=1).astype(int)

        P1.mu_r_1 = np.mean(in_1, axis=1).astype(int)

        P2.mu_r_2 = np.mean(in_2, axis=1).astype(int)

        # RSS shares for each parties
        P0.mu_r_1 = np.copy(P1.mu_r_1)

        P1.mu_r_2 = np.copy(P2.mu_r_2)

        P2.mu_r_0 = np.copy(P0.mu_r_0)

        return P0.mu_r_0, P1.mu_r_1, P2.mu_r_2


    def secResidue(self, P0, P1, P2, in_0, in_1, in_2, mu_ij_0, mu_r_0, mu_c_0, mu_ij_1, mu_r_1, mu_c_1,
                   mu_ij_2, mu_r_2, mu_c_2):
        """Secured residue based on RSS consisting of local linear functions."""
        # Find residue with given mean values
        P0.r_ij_0 =  in_0 - mu_r_0[:, np.newaxis] - mu_c_0 + mu_ij_0

        P1.r_ij_1 =  in_1 - mu_r_1[:, np.newaxis] - mu_c_1 + mu_ij_1

        P2.r_ij_2 =  in_2 - mu_r_2[:, np.newaxis] - mu_c_2 + mu_ij_2

        # RSS shares for each parties
        P0.r_ij_1  = np.copy(P1.r_ij_1)

        P1.mu_ij_2 = np.copy(P2.r_ij_2)

        P2.mu_ij_0 = np.copy(P0.r_ij_0)

        return P0.r_ij_0, P1.r_ij_1, P2.r_ij_2


    def secSquaring(self, P0, P1, P2, r_ij_0, r_ij_1, r_ij_2):
        """Secured multiplication based on RSS for doing squaring."""
        # Square the residue for each party with given inputs
        P0.r2_ij_0 =  (r_ij_0 * r_ij_0) + (r_ij_1 * r_ij_0) + (r_ij_0 * r_ij_1)

        P1.r2_ij_1 =  (r_ij_1 * r_ij_1) + (r_ij_2 * r_ij_1) + (r_ij_1 * r_ij_2)

        P2.r2_ij_2 =  (r_ij_2 * r_ij_2) + (r_ij_0 * r_ij_2) + (r_ij_2 * r_ij_0)

        # Find the structure of multiplicants
        num_rows, num_cols = r_ij_0.shape

        # Generate random shares
        P0.share_0 = self.gen_RandSharing(num_rows, num_cols)

        P1.share_1 = self.gen_RandSharing(num_rows, num_cols)

        P2.share_2 = self.gen_RandSharing(num_rows, num_cols)

        # Zero shares for each party
        P0.c_0 = self.gen_ZeroSharing(P0.share_0, P1.share_1)

        P1.c_1 = self.gen_ZeroSharing(P1.share_1, P2.share_2)

        P2.c_2 = self.gen_ZeroSharing(P2.share_2, P0.share_0)

        # RSS shares for each party
        P0.r2_ij_1 = np.copy(P1.r_ij_1) + P0.c_0

        P1.r2_ij_2 = np.copy(P2.r_ij_2) + P1.c_1

        P2.r2_ij_0 = np.copy(P0.r_ij_0) + P2.c_2

        return P0.r2_ij_0, P1.r2_ij_1, P2.r2_ij_2


    def gen_RandSharing(self, num_rows, num_cols):
        """Generation of random numbers from an unsigned ring."""
        rng     = np.random.default_rng(seed=42)
        rshare  = rng.integers(0, self.highest_range, size=(num_rows, num_cols), dtype="int64")

        return rshare


    def gen_ZeroSharing(self, current_share, next_share):
        """Generation of zero shares from rng for RSS resharing."""
        zero_share = current_share - next_share

        return  zero_share

    def fss_evaluation(self, share_0, share_1, in_len, sdel):
        """FSS IC Sign Evaluation"""
        # Inputs and parameters e.g. threshold
        gamma = 0
        z_0 = share_0.astype(funshade.DTYPE)
        z_1 = share_1.astype(funshade.DTYPE)

        # Check if the input length is None or not (it can be length of the input or one)
        if in_len is None:
            K = len(z_0)
        else:
            K = in_len

        # Create parties
        class party:
            def __init__(self, j: int):
                self.j = j

        P0 = party(0)
        P1 = party(1)

        # Generate setup preprocessing materials
        r_in0, r_in1, k0, k1 = funshade.FssGenSign(K, gamma)

        P0.r_in_j = r_in0
        P1.r_in_j = r_in1
        P0.k_j = k0
        P1.k_j = k1

        # Send the shares to the parties
        P0.z_j = z_0
        P1.z_j = z_1

        # Mask the public input to FSS gate
        P0.z_hat_j = P0.z_j + P0.r_in_j
        P1.z_hat_j = P1.z_j + P1.r_in_j

        P1.z_hat_nj = P0.z_hat_j
        P0.z_hat_nj = P1.z_hat_j

        # Evaluation with FSS IC gate
        P1.o_j = funshade.eval_sign(K, P1.j, P1.k_j, P1.z_hat_j, P1.z_hat_nj)
        P0.o_j = funshade.eval_sign(K, P0.j, P0.k_j, P0.z_hat_j, P0.z_hat_nj)

        # Outputs the results when in_len is not None or sdel is true
        if in_len is None or sdel:
            return P0.o_j, P1.o_j
        # Construct the output of both parties
        else:
            o = P0.o_j + P1.o_j
            return o


    def fss_evaluation_without_len(self, share_0, share_1):
        """FSS Sign Evaluation without having length of input vector."""
        # Input parameters threshold, and length of matrix
        gamma = 0
        z_0 = share_0.astype(funshade.DTYPE)
        z_1 = share_1.astype(funshade.DTYPE)
        K = len(z_0)

        # Create parties
        class party:
            def __init__(self, j: int):
                self.j = j

        P0 = party(0)
        P1 = party(1)

        # Generate setup preprocessing materials
        r_in0, r_in1, k0, k1 = funshade.FssGenSign(K, gamma)

        P0.r_in_j = r_in0
        P1.r_in_j = r_in1
        P0.k_j = k0
        P1.k_j = k1

        # Send the shares to the parties
        K = len(z_0)
        P0.z_j = z_0
        P1.z_j = z_1

        # Mask the public input to FSS gate
        P0.z_hat_j = P0.z_j + P0.r_in_j
        P1.z_hat_j = P1.z_j + P1.r_in_j

        P1.z_hat_nj = P0.z_hat_j
        P0.z_hat_nj = P1.z_hat_j

        # Evaluation with FSS IC gate
        P1.o_j = funshade.eval_sign(K, P1.j, P1.k_j, P1.z_hat_j, P1.z_hat_nj)
        P0.o_j = funshade.eval_sign(K, P0.j, P0.k_j, P0.z_hat_j, P0.z_hat_nj)

        return P0.o_j, P1.o_j


    def fss_evaluation_sdel(self, share_0, share_1, len, t_size):
        """FSS IC Sign Evaluation when having known length of input vector particularly for single node deletion"""
        # Input parameters threshold, and length of matrix
        gamma = 0
        z_0 = share_0.astype(funshade.DTYPE)
        z_1 = share_1.astype(funshade.DTYPE)
        K = len

        # Create parties
        class party:
            def __init__(self, j: int):
                self.j = j

        P0 = party(0)
        P1 = party(1)

        # Generate setup preprocessing materials
        r_in0, r_in1, k0, k1 = funshade.FssGenSign(K, gamma)

        P0.r_in_j = r_in0
        P1.r_in_j = r_in1
        P0.k_j = k0
        P1.k_j = k1

        # Send the shares to the parties
        P0.z_j = z_0
        P1.z_j = z_1

        # Mask the public input to FSS gate
        P0.z_hat_j = P0.z_j + P0.r_in_j
        P1.z_hat_j = P1.z_j + P1.r_in_j

        P1.z_hat_nj = P0.z_hat_j
        P0.z_hat_nj = P1.z_hat_j

        with open('result_size.txt', 'w') as saveFile:
            saveFile.write(str(P0.z_hat_j) + "\n")
            saveFile.write(str(P1.z_hat_j) + "\n")
        t_size.append(os.path.getsize("result_size.txt"))

        # Evaluation with FSS IC gate
        P1.o_j = funshade.eval_sign(K, P1.j, P1.k_j, P1.z_hat_j, P1.z_hat_nj)
        P0.o_j = funshade.eval_sign(K, P0.j, P0.k_j, P0.z_hat_j, P0.z_hat_nj)

        return P0.o_j, P1.o_j


    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.msr_threshold != 'estimate' and self.msr_threshold < 0.0:
            raise ValueError("msr_threshold must be equal to 'estimate' or a numeric value >= 0.0, got {}".format(self.msr_threshold))

        if self.multiple_node_deletion_threshold < 1.0:
            raise ValueError("multiple_node_deletion_threshold must be >= 1.0, got {}".format(self.multiple_node_deletion_threshold))

        if self.data_min_cols < 100:
            raise ValueError("data_min_cols must be >= 100, got {}".format(self.data_min_cols))















