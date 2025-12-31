library(ggplot2)


plot_ns_matrix <- function(
  mat,
  counties,
  fips_in_order = NULL,      # named character: names = counties, values = FIPS (recommended)
  fips_to_county = NULL,     # named character: names = FIPS, values = county label (alternative)
  show_code = FALSE,         # if TRUE, label axes as "FIPS County"
  digits = 2,
  text_size = 5,
  axis_text_size = 12,
  tile_color = "grey80",
  tile_linewidth = 0.3,
  flip_color = FALSE,
  color_scale = 1.0
) {
  stopifnot(is.matrix(mat) || inherits(mat, "Matrix"))
  stopifnot(nrow(mat) == ncol(mat))
  stopifnot(is.character(counties), length(counties) == nrow(mat))

  # Build FIPS -> county mapping
  if (!is.null(fips_in_order)) {
    stopifnot(is.character(fips_in_order), !is.null(names(fips_in_order)))
    stopifnot(all(counties %in% names(fips_in_order)))
    fips_to_county <- setNames(names(fips_in_order), unname(fips_in_order))
    county_to_fips <- fips_in_order[counties]
  } else if (!is.null(fips_to_county)) {
    stopifnot(is.character(fips_to_county), !is.null(names(fips_to_county)))
    # If mat has FIPS rownames/colnames, infer order from counties via fips_to_county
    if (!is.null(rownames(mat))) {
      # Map counties -> fips by matching labels
      county_to_fips <- vapply(
        counties,
        function(cty) {
          hits <- names(fips_to_county)[fips_to_county == cty]
          if (length(hits) == 0) NA_character_ else hits[1]
        },
        character(1)
      )
    } else {
      county_to_fips <- rep(NA_character_, length(counties))
    }
  } else {
    # No mapping provided: just use counties directly
    fips_to_county <- setNames(counties, counties)
    county_to_fips <- counties
  }

  # Convert matrix to long df
  df <- as.data.frame(as.table(mat))
  names(df) <- c("row", "col", "val")
  df$row <- as.character(df$row)
  df$col <- as.character(df$col)

  # Default: if mat row/col names are missing, assume they correspond to `counties`
  if (is.null(rownames(mat)) || is.null(colnames(mat))) {
    # Create pseudo codes based on counties order
    codes <- if (!is.null(fips_in_order)) unname(fips_in_order[counties]) else counties
    rownames(mat) <- codes
    colnames(mat) <- codes
    df <- as.data.frame(as.table(mat))
    names(df) <- c("row", "col", "val")
    df$row <- as.character(df$row)
    df$col <- as.character(df$col)
  }

  # Axis labels: map code -> county, optionally prepend code
  row_lab <- fips_to_county[df$row]
  col_lab <- fips_to_county[df$col]

  if (show_code) {
    row_lab <- paste0(df$row, " ", row_lab)
    col_lab <- paste0(df$col, " ", col_lab)
    axis_levels <- if (!is.null(fips_in_order)) {
      paste0(unname(fips_in_order[counties]), " ", counties)
    } else {
      unique(col_lab)
    }
  } else {
    axis_levels <- counties
  }

  df$row_lab <- factor(row_lab, levels = axis_levels)
  df$col_lab <- factor(col_lab, levels = axis_levels)

  p = ggplot(df, aes(x = col_lab, y = row_lab, fill = val)) +
    geom_tile(color = tile_color, linewidth = tile_linewidth)
    
  if (flip_color) {
    p = p +
      geom_text(
        aes(label = sprintf(paste0("%.", digits, "f"), val)),
        size = text_size,
        color = ifelse(df$val > 0.5 * color_scale, "black", "white")
      ) +
      scale_fill_gradient(limits = c(0, color_scale), low = "black", high = "white")
  } else {
     p = p +
      geom_text(
        aes(label = sprintf(paste0("%.", digits, "f"), val)),
        size = text_size,
        color = ifelse(df$val > 0.5 * color_scale, "white", "black")
      ) +
      scale_fill_gradient(limits = c(0, color_scale), low = "white", high = "black")
  }
  
  p = p +
    coord_equal() +
    guides(fill = "none") +
    scale_x_discrete(position = "top", expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0), limits = rev(axis_levels)) +
    theme_void() +
    theme(
      axis.text.x.top = element_text(size = axis_text_size, angle = 45, hjust = +0.5),
      axis.text.y     = element_text(size = axis_text_size, angle = 90),
      axis.title      = element_blank(),
      axis.ticks      = element_blank(),
      plot.margin     = margin(0.5, 0.5, 0.5, 0.5)
    )
  
  return(p)
}



plot_ts_ci_multi = function(psi_list = NULL,
                             fontsize = 20,
                             time_label = NULL,
                             legend.position = "none",
                             legend.nrow = 1,
                             legend.name = "Method",
                             xlab = "Time",
                             ylab = expression(psi[t]),
                             alpha = 0.2) {
  mlist_external = toupper(c("EpiEstim", "WT", "Koyama2021"))
  clist_external = c("seagreen", "orange", "burlywood4")

  mlist_dgtf = toupper(c(
    "LBA", "LBE", "LBA.DF", "LBA.W", "HS", "Apparent",
    "TFS", "TFS.W", "TFS.DF", "HS.EFF", "Local",
    "HVB", "HVA", "Import",
    "MCMC", "Hawkes",
    "True",
    "VB",
    "PL",
    "SMCS-FL", "MCS",
    "APF",
    "SMCF-BF",
    "SMCS-BS", "BS", "FFBS",
    "Exponential", "lognorm", "Discretized Hawkes",
    "Softplus", "nbinom", "Distributed Lags"
  ))
  clist_dgtf = c(
    rep("maroon", 6), # LBA, LBE, LBA.DF, LBA.W
    rep("peru", 5), # TFS
    rep("purple", 3), # HVB, HVA **
    rep("darkturquoise", 2), # MCMC
    "black", # True
    "salmon", # VB
    "royalblue", # PL **
    rep("cornflowerblue", 2), # SMCS-FL, MCS **
    "gold", # APF
    "sandybrown", # SMCF-BF
    rep("mediumaquamarine", 3), # SMCS-BS
    rep("maroon", 3),
    rep("royalblue", 3)
  )

  mlist = c(mlist_external, mlist_dgtf)
  clist = c(clist_external, clist_dgtf)


  nl = length(psi_list)
  n = max(c(unlist(lapply(psi_list, function(psi) {
    dim(psi)[1]
  }))))

  if (is.null(time_label)) {
    time_label = c(1:n)
  } else {
    time_label = time_label[1:n]
  }


  for (i in 1:nl) {
    if (toupper(names(psi_list)[i]) == toupper("Koyama2021")) {
      psi_list[[i]] = psi_list[[i]][-1, 1:3]
    }

    psi_list[[i]] = data.frame(
      lobnd = psi_list[[i]][, 1],
      est = psi_list[[i]][, 2],
      hibnd = psi_list[[i]][, 3],
      time = time_label[(n - dim(psi_list[[i]])[1] + 1):n]
    )
    psi_list[[i]]$method = rep(toupper(names(psi_list)[i]), dim(psi_list[[i]])[1])
  }

  psi_list = do.call(rbind, psi_list)
  psi_list = as.data.frame(psi_list)


  ####
  methods = unique(psi_list$method)
  # cols = sapply(methods,function(m,mlist){which(m==mlist)},mlist)
  col_tmp = NULL
  cols = NULL
  for (m in methods) {
    if (m %in% mlist) {
      ccctmp = which(m == mlist)
      col_tmp = c(col_tmp, ccctmp)
      cols = c(cols, clist[ccctmp])
    } else {
      tmp = setdiff(colors(), unique(mlist))[-c(145:372)]
      cc = sample(tmp, 1)
      cols = c(cols, cc)
    }
  }

  # cols = clist[col_tmp]

  p = ggplot(psi_list, aes(x = time, y = est, group = method)) +
    geom_ribbon(aes(ymin = lobnd, ymax = hibnd, fill = method),
      alpha = alpha, na.rm = TRUE
    ) +
    geom_line(aes(color = method), na.rm = TRUE) +
    scale_color_manual(name = legend.name, breaks = methods, values = cols) +
    scale_fill_manual(name = legend.name, breaks = methods, values = cols) +
    theme_minimal() +
    xlab(xlab) +
    theme(
      text = element_text(size = fontsize),
      legend.position = legend.position
    ) +
    guides(colour = guide_legend(nrow = legend.nrow)) +
    ylab(ylab)


  return(p)
} # plot_ts_ci_multi



plot_ts_ci_single = function(
    psi_quantile, psi_true = NULL,
    main = "Posterior",
    ylab = "psi",
    show_caption = FALSE) {
  ymin = min(c(psi_quantile))
  ymax = max(c(psi_quantile))

  dat = data.frame(
    psi = psi_quantile[, 2],
    psi_min = psi_quantile[, 1],
    psi_max = psi_quantile[, 3],
    time = 0:(dim(psi_quantile)[1] - 1)
  )

  if (!is.null(psi_true)) {
    dat$true = psi_true
  }

  p = ggplot(data = dat, aes(x = time, y = psi)) +
    theme_minimal() +
    geom_ribbon(aes(ymin = psi_min, ymax = psi_max), alpha = 0.5, fill = "lightgrey") +
    geom_line(na.rm = TRUE) +  
    ylab(ylab) +
    labs(title = main)

  if (!is.null(psi_true)) {
    p = p + geom_line(
      aes(y = true, x = time),
      data = dat,
      color = "maroon", alpha = 0.8
    )
  }

  if (show_caption) {
    p = p + labs(caption = "Red line represents true values, black line is the posterior median with a credible interval from the 2.5% quantile to 97.5% quantile.")
  }

  return(p)
} # plot_ts_ci_single
