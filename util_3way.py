#!/usr/bin/env python

"""util_3way.py"""


import sys

import numpy as np
import scipy.stats as st
import statsmodels
import statsmodels.api as sm 

THRESHOLD = 0.05


def squeeze_var(var, df):
    '''Empirical Bayes posterior variances, based on LIMMA's squeezeVar.R
    (last modified 2 Dec 2013).
    Smooth a set of sample variances by computing empirical Bayes posterior means. 
    
    The sample variances var are assumed to follow scaled chi-squared distributions. 
    An inverse chi-squared prior is assumed for the true variances. 
    The scale and degrees of freedom for the prior distribution are estimated from the data.

    The effect of this function is to smooth or shrink the variances towards a common value. 
    The smoothed variances have a smaller expected mean square error to the true variances
    than do the sample variances themselves.

 
    Input
    -----
    var: list of floats
        List of independent sample variances.
    
    df: {int, list of ints}
        List of degrees of freedom for the sample variances.
        If only one value, same for all variances.
    
    
    Output
    ------
    var_post:  list of floats
       List of of posterior variances.
       Corresponds to s2_post.
    
    var_prior:  list of floats
       Location of prior distribution.
       Corresponds to s2_prior.
       
    df_prior: int
        Degrees of freedom of prior distribution.
    '''
    n = var.shape[0]
    if n == 1:
        return [var, var, 0]
    #if len(df) == 1:
    #    df = [df for idx in range(n)]
    #if len(df) != n:
    #    sys.stderr.write("lengths of df and var differ.\n")
    #    sys.exit(-1)
        
    # Estimate prior variance and df:     
    # parameters of a scaled F-distribution 
    # given the first degrees of freedom
    [df0, df_prior, var_prior, loc] = st.f.fit(var, f0=df, floc=0)
    
    # Squeeze the posterior variances
    df_total = df + df_prior
    var_post = (df * var + df_prior * var_prior) / df_total
    
    return [var_post, var_prior, df_prior]



def moderated_ttest(design, contrast, expression):
    """
    Compute moderated t-test comparison in the spirit of LIMMA.

    Input
    -----
    design: (experiments, conditions) np.array
        Design matrix.
    contrast: (conditions, comparisons) np.array
        Contrast matrix
    expression: (num_genes, experiments) np.array
        Expression levels.

    Output
    -----
    mt_test: (num_genes, comparisons) np.array
        p-values of comparisons.

    mt_test_comp: (num_genes, comparisons) np.array
        Outcome of comparison:
        -1 if < 
         0 if =
        +1 if >     
    """
    num_genes = expression.shape[0]
    # Add 1s to design for intercept
    ones = np.ones((design.shape[0], 1))
    design_intercept = np.hstack((design, ones)) 

    # Fit alphas 
    lm_fits = []
    for gene_idx in range(num_genes):
        fitr = sm.OLS(expression[gene_idx, :], design_intercept)
        res = fitr.fit()
        lm_fits.append(res)
    alphas_intercept = np.array([res.params for res in lm_fits])

    # Remove intercept
    alphas = alphas_intercept[:, :-1]

    # Compute beta coefficients
    betas = np.dot(alphas, contrast)

    # Compute standard deviation of the alphas
    C = np.linalg.inv(np.dot(design.T, design))
    stdev_unscaled_alpha = np.sqrt(np.diag(C))

    # Compute unadjusted standard deviations of the betas
    cov = np.dot(np.dot(contrast.T, C), contrast)     
    usd = np.sqrt(np.diag(cov))

    # Check whether the design matrix was orthogonal: is C close to identity?
    orthog = np.alltrue(np.abs(C-np.eye(C.shape[0])) < 1e-10)
    
    # Compute residual standard deviations, squared
    sigma2 = np.array([res.mse_resid for res in lm_fits])

    # Compute residual degrees of freedom
    df_resid = np.array([res.df_resid for res in lm_fits])
    if np.shape(np.unique(df_resid))[0] != 1:
        sys.stderr.write("More than one df??\n")
        sys.exit(-1)
    df_resid = df_resid[0]

    # Squeeze variance
    [s2_post, s2_prior, df_prior] = squeeze_var(sigma2, df_resid)

    # Compute t-tests
    mt_test = np.zeros(betas.shape) # p-values
    mt_test_comp = np.zeros(mt_test.shape, dtype=int) # -1 if comparison is <0, +1 if it is >0, 0 if it is=

    for gene_idx in range(num_genes):
        # beta coefficients
        coeffs = betas[gene_idx, :]     
    
        # standard deviations of the beta coefficients
        if orthog:
            stdev_unscaled = usd
        else:     
            # Standard deviations of the betas need adjustment
            # Compute covariance matrix of the alphas
            res = lm_fits[gene_idx]
            cov_mat = res.mse_resid * C
    
            # Compute correlation matrix of the alphas
            cormatrix = statsmodels.stats.moment_helpers.cov2corr(cov_mat)
        
            # Adjust standard deviations of the betas
            R = np.linalg.cholesky(cormatrix)
            RUC = np.dot(R, np.dot(np.diag(stdev_unscaled_alpha), contrast))
            stdev_unscaled = np.sqrt(np.dot(np.ones((alphas.shape[1], )), RUC**2))
    
        # t-test
        t = coeffs / stdev_unscaled / np.sqrt(s2_post[gene_idx])
        df = df_resid + df_prior
        
        # p-value
        pvals = 2 * st.t.cdf(-np.abs(t), df=df)
        mt_test[gene_idx, :] = pvals
            
    # Correct for multiple hypotheses testing
    pvals_f = mt_test.flatten()
    mt_test = statsmodels.sandbox.stats.multicomp.multipletests(pvals_f, method='fdr_bh')[1]
    mt_test = mt_test.reshape(mt_test_comp.shape)

    # Final comparison
    for gene_idx in range(num_genes):
        for comp_idx in np.where(mt_test[gene_idx, :] < THRESHOLD)[0]:
            b = np.mean(expression[gene_idx,
                                   np.where(design[:, list(contrast[:, comp_idx]).index(-1)])[0]])
            a = np.mean(expression[gene_idx,
                                   np.where(design[:, list(contrast[:, comp_idx]).index(1)])[0]])
            mt_test_comp[gene_idx, comp_idx] = np.sign(a - b)
        
    return sigma2, mt_test, mt_test_comp 


def compute_comp_array(comp_vec):
    """ Compute array of 8x8 pairwise comparisons
    from vector of 28 pairwise comparisons.
    
    Input
    -----
    comp_vec: (28, ) int array
       Pairwise comparisons, represented as -1/0/+1.
       
    Output
    ------
    comp_array: (8, 8) int array
        comp_array[i, j] is +1 if value[i] < value[j], 
                             0 if value[i] = value[j],
                            -1 if value[i] > value[j].
    """
    comp_array = np.zeros((8, 8), dtype=int)
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            comp_array[i, j] = int(comp_vec[idx])
            comp_array[j, i] = - comp_array[i, j]
            idx += 1
    return comp_array

    
def compute_comp_array_pval(comp_vec):
    """ Compute array of 8x8 pairwise comparisons
    from vector of 28 pairwise comparisons.
    
    Input
    -----
    comp_vec: (28, ) int array
       Pairwise comparisons, represented by their p-value
       
    Output
    ------
    comp_array: (8, 8) int array
        comp_array[i, j] is -np.log10(p-value of comparison of i, j)
    """
    comp_array = np.zeros((8, 8))
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            comp_array[i, j] = -np.log10(comp_vec[idx])
            comp_array[j, i] = - comp_array[i, j]
            idx += 1
    return comp_array

    


def compute_levels(comp_array):
    """ Compute levels from 8x8 array of pairwise comparisons.
    
    Levels are integers between 0 and at most 7. 
    There are as many levels as significantly different expression values,
    and they are ordered, with no gaps.
    
    E.g.
    [0, 1, 4, 3, 2, 2, 2, 2] is a valid vector of levels.
    [0, 1, 5, 3, 2, 2, 2, 2] is not.
    
    Input
    ----- 
    comp_array: (8, 8) int array
        comp_array[i, j] is +1 if value[i] < value[j], 
                             0 if value[i] = value[j],
                            -1 if value[i] > value[j].
       
    Output
    ------
    levels: (8, ) int array
        Levels of 0, A, B, C, AB, AC, BC, ABC.
    """  
    # Bubble-sort the 8 values
    sorted_list = [0]
    for i in range(1, 8):
        # put i in its place:
        for j, jval in enumerate(sorted_list):
            if comp_array[i, jval] > 0: # value[i] < value[j] 
                # insert i before position j
                sorted_list.insert(j, i) 
                break
        # insert i at end of list (if not inserted before)
        if len(sorted_list) < (i+1):
            sorted_list.append(i)
            
    # Assign levels. We'll start left so from the highest level.    
    levels = np.ones(len(sorted_list), dtype=int) * 11
    max_level = 0
    levels[sorted_list[0]] = max_level
    for i in range(1, len(sorted_list)):
        if comp_array[sorted_list[i-1], sorted_list[i]] > 0:
             max_level += 1         
        levels[sorted_list[i]] = max_level    
    return levels


def compute_XYZ_map(comp_array):
    """ Assign X, Y and Z from 8x8 array of pairwise comparisons. 
    
    Reorder the conditions A, B, C in such a way that
    X <= Y <= Z
    if X=Y then XZ <= YZ
    if Y=Z then XY <= XZ
    if X=Y=Z then XY <= XZ <= YZ.
    
    Input
    -----
    comp_array: (8, 8) int array
        comp_array[i, j] is +1 if value[i] < value[j], 
                             0 if value[i] = value[j],
                            -1 if value[i] > value[j].   
       
    Output
    ------
    map: list of length 8
       Mapping vector such that expression_XYZ[i] = expression_ABC[map[i]]
    """
    map = range(8)
    # if A<B
    if comp_array[1, 2] > 0:
        # if B<C: do not change anything
        if comp_array[2, 3] < 0:
            # if B>C:
            #print "Z=B and XY=AC"
            map[3] = 2
            map[4] = 5 
            # if A<C
            if comp_array[1, 3] > 0:
                #print "X=A, Y=C, Z=B"
                map[2] = 3
                map[5] = 4 #AB 
            elif comp_array[1, 3] < 0:
                #print "X=C, Y=A, Z=B"
                map[1] = 3
                map[2] = 1
                map[5] = 6 # XZ = BC
                map[6] = 4 # YZ = AB
            else: # i.e. A=C
                # if AB <= CB
                if comp_array[4, 6] >= 0:
                    #print "X=A, Y=C"
                    map[2] = 3
                    map[5] = 4 #AB
                else:
                    #print "X=C, Y=A"
                    map[1] = 3
                    map[2] = 1
                    map[5] = 6 #BC
                    map[6] = 4
        elif comp_array[2, 3] == 0:
            # X=A, do not change map[1]
            # if AB <= AC, do not change anything
            if comp_array[4, 5] < 0:
                #print "Y=C, Z=B"
                map[2] = 3
                map[3] = 2
                map[4] = 5 #AC
                map[5] = 4 #AB
    elif comp_array[1, 2] < 0:
        # if B>C
        if comp_array[2, 3] < 0:
            #print "X=C, Y=B, Z=A"
            map[1] = 3
            map[2] = 2
            map[3] = 1
            map[4] = 6 #BC
            map[5] = 5 #AC
            map[6] = 4
        # elif B<C
        elif comp_array[2, 3] < 0:
            #print "X=B and YZ=AC"
            map[1] = 2            
            map[6] = 5
            # if A<C
            if comp_array[1, 3] > 0:
                #print "X=B, Y=A, Z=C"
                map[2] = 1
                map[5] = 6 #BC
            # elif A>C
            elif comp_array[1, 3] < 0:
                #print "X=B, Y=C, Z=A"
                map[2] = 3
                map[3] = 1
                map[4] = 6 #BC
                map[5] = 4 #AB
            else: # i.e. A=C
                # if AB <= BC
                if comp_array[4, 6] >= 0:
                    #print "Y=A, Z=C"
                    map[2] = 1
                    map[5] = 6 #BC
                else:
                    #print "Y=C, Z=A"
                    map[2] = 3
                    map[3] = 1
                    map[4] = 6 #BC
                    map[5] = 4 #AB
        else: # i.e. B=C
            #print "Z=A and XY=BC"
            map[3] = 1
            map[4] = 6             
            # if AB <= BC:
            if comp_array[4, 6] >= 0:
                #print "X=B, Y=C"
                map[1] = 2
                map[2] = 3
                map[5] = 4 #AB
                map[6] = 5
            else:
                #print "X=C, Y=B"
                map[1] = 3
                map[6] = 4 #AB
    return map


def compute_delta_profiles(mt_test_comp):
    """
    Compute delta-profiles based on re-ordered levels (compared to control).

    Input:
    -----
    mt_test_comp: (num_genes, comparisons) np.array
        Outcome of comparison:
        -1 if < 
         0 if =
        +1 if >     

    Ouptut:
    ------
    delta_test_comp: (num_genes, conditions-1) np.array
        Delta profiles of the genes.
        Difference in level with the control.
        Conditions are re-ordered so that the individual condition with smallest effect is first.
    """
    delta_profiles = np.zeros((mt_test_comp.shape[0], 7), dtype=int)

    for gene_idx in range(mt_test_comp.shape[0]):
        comp_array = compute_comp_array(mt_test_comp[gene_idx, :])
        comp_array = make_transitive(comp_array) # NEW
        levels = np.array(compute_levels(comp_array))
        xyz_map = compute_XYZ_map(comp_array)
        levels_reordered = levels[xyz_map]
        delta_profiles[gene_idx, :] = (levels_reordered-levels_reordered[0])[1:]

    return delta_profiles


def understand_f_fitting(sigma2):
    """
    Test function: Understanding the F scaled fitting procedure.
    
    """
    import matplotlib.pyplot as plt
    
    prms = st.f.fit(sigma2, f0=19)#, floc=0)
    print prms
    dfn = prms[0]
    dfd = prms[1]
    scale_ = prms[3]
    loc_ = prms[2]
    x = np.linspace(st.f.ppf(0.01, dfn, dfd, scale=scale_, loc=loc_),
                    st.f.ppf(0.99, dfn, dfd, scale=scale_, loc=loc_), 100)
    
    rv = st.f(dfn, dfd, scale=scale_)#, loc=loc_)
    plt.plot(x, rv.pdf(x), color='#ee9041', lw=2)
    h = plt.hist(sigma2, normed=True, color='#459db9')


def make_transitive(comp_array):
    """ Make a comparison array transitive
    by correcting significant into non-significant when inconsistent.

    Input
    -----
    comp_array: (8, 8) int array
        comp_array[i, j] is +1 if value[i] < value[j], 
                             0 if value[i] = value[j],
                            -1 if value[i] > value[j].

    Output
    ------
    trans_comp_array: (8, 8) int array
        such that if (1) = (2) and (2) = (3) but (1) < (3)
        then (2) < (3) after all.
    """
    n = comp_array.shape[0]
    trans_comp_array = np.array(comp_array)
    for idx_1 in range(n):
        for idx_2 in range(idx_1+1, n):
            if (comp_array[idx_1, idx_2] == 0):
                # idx_1 and idx_2 are identical
                trans_comp_array[:idx_2, idx_2] = trans_comp_array[:idx_2, idx_1]
            
            
    for idx_1 in range(n):
        for idx_2 in range(idx_1+1, n):
            trans_comp_array[idx_2, idx_1] = - trans_comp_array[idx_1, idx_2]

    return trans_comp_array
    