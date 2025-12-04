library(dplyr)

## Helper: standardize FIPS to 5-character strings
std_fips = function(x) sprintf("%05d", as.integer(x))

## Helper: build an S x S matrix from an edge list (origin, dest, value)
edge_to_mat = function(df, origin_col, dest_col, value_col, counties, symmetric = FALSE) {
  county_ids = std_fips(counties)
  df2 = df %>%
    dplyr::mutate(
      origin = factor(std_fips(.data[[origin_col]]), levels = county_ids),
      dest   = factor(std_fips(.data[[dest_col]]),   levels = county_ids)
    )
  
  mat = xtabs(
    formula = as.formula(paste(value_col, "~ dest + origin")),
    data = df2
  )
  
  # ensure full SxS with correct ordering
  mat = mat[county_ids, county_ids, drop = FALSE]
  as.matrix(mat)

  if (symmetric) {
    mat[is.na(mat)] = 0
    mat = pmax(mat, t(mat))
    diag(mat) = 0
  }

  return(mat)
}

## Helper: rescale positive matrix to [0, 1]
scale_to_01 = function(mat) {
  m = max(mat, na.rm = TRUE)
  if (!is.finite(m) || m <= 0) {
    return(matrix(0, nrow(mat), ncol(mat), dimnames = dimnames(mat)))
  }
  mat / m
}

## Helper: safe log1p transform then rescale to [0,1]
log1p_scale = function(mat) {
  nz = mat > 0
  out = matrix(0, nrow(mat), ncol(mat), dimnames = dimnames(mat))
  if (!any(nz)) return(out)
  
  x = log1p(mat[nz])
  out[nz] = (x - min(x)) / (max(x) - min(x))
  out
}


generate_commute_matrix = function(commute_raw, county_info) {
  county_ids = std_fips(county_info$county_fips)
  pop_vec = county_info$population[match(std_fips(county_ids), std_fips(county_info$county_fips))]
  names(pop_vec) = county_ids
  pop_vec[!is.finite(pop_vec) | pop_vec <= 0] = NA

  commute_df = commute_raw
  commute_df$fips_from = county_info$county_fips[match(commute_df$county_from, county_info$county)]
  commute_df$fips_to   = county_info$county_fips[match(commute_df$county_to, county_info$county)]

  commute_mat = edge_to_mat(
    df = commute_df,
    origin_col = "fips_from",
    dest_col   = "fips_to",
    value_col  = "count",
    counties   = county_ids
  )

  commute_pc = sweep(commute_mat, 2, pop_vec, "/")
  commute_pc[!is.finite(commute_pc)] = 0
  commute_pc_filled = commute_pc

  for (k in seq_len(ncol(commute_pc_filled))) {
    off = commute_pc_filled[-k, k]  # off-diagonal per-capita flows from origin k
    if (all(off == 0)) {
      # completely isolated in census flow; give a small positive diagonal
      commute_pc_filled[k, k] = 0
    } else {
      commute_pc_filled[k, k] = max(off, na.rm = TRUE)
    }
  }
  
  return(commute_pc_filled)
}
