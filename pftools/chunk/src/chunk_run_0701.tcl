#
# Write to the Parameter file:
#
set fileId [open $par_file w 0600]
if {$plot_landscape == "yes"} {
  puts $fileId "1"
} else {
  puts $fileId "0"
}
puts $fileId "$plot_translation_x $plot_translation_y"
puts $fileId "$plot_textsize"
puts $fileId "[winfo rgb . $plot_background_color] \
              [winfo rgb . $plot_text_color] \
              [winfo rgb . $plot_axes_color]" 
puts $fileId "$chunk_n_x $chunk_n_y $chunk_n_z"
puts $fileId "$chunk_space_x $chunk_space_y $chunk_space_z"
set nx $chunk_n_x
set ny $chunk_n_y
set nxy [expr $nx * $ny]
set chunk_erase_list " "
set chunk_erase_n 0
for {set k 1} {$k <= $chunk_n_z} {incr k 1} {
for {set j 1} {$j <= $ny} {incr j 1} {
for {set i 1} {$i <= $nx} {incr i 1} {
 if {$chunk_num($i,$j,$k) == 1} {
  set chunk_erase_n [expr $chunk_erase_n + 1]
  lappend chunk_erase_list [expr  $nxy * ($k - 1)  + $nx * ($j - 1) + $i] 
 }
}
}
}
if {$chunk_view != "3D"} {set chunk_erase_n 0}
puts $fileId "$chunk_erase_n"
if {$chunk_erase_n == 0} {
  puts $fileId "0"
} else {
  puts $fileId $chunk_erase_list
}
puts $fileId "$ps_file"
if {$ps_file_format == "eps"} {
  puts $fileId "1"
  puts $fileId "$eps_BoundingBox_min_x $eps_BoundingBox_min_y $eps_BoundingBox_max_x $eps_BoundingBox_max_y"
  puts $fileId "$eps_creator"
  puts $fileId "$eps_date"
  puts $fileId "$eps_time"
} else {
  puts $fileId "0"
} 
puts $fileId "$block_min_x $block_min_y $block_min_z"
puts $fileId "$block_d_x $block_d_y $block_d_z"
puts $fileId "$chunk_crop_x_min $chunk_crop_x_max $chunk_crop_y_min $chunk_crop_y_max $chunk_crop_z_min $chunk_crop_z_max"
puts $fileId "$block_scale_x $block_scale_y $block_scale_z"
puts $fileId "$label_space_x $label_space_y $label_space_z"
puts $fileId "$label_decimals_x $label_decimals_y $label_decimals_z"
puts $fileId "$label_tic_per_x $label_tic_per_y $label_tic_per_z"
puts $fileId "$label_x"
puts $fileId "$label_y"
puts $fileId "$label_z"
puts $fileId "$plot_title_1"
puts $fileId "$plot_title_2"
puts $fileId "$plot_title_translation_x $plot_title_translation_y"
if {$plot_title_justification == "left"} {puts $fileId "0"}
if {$plot_title_justification == "center"} {puts $fileId "1"}
if {$plot_title_justification == "right"} {puts $fileId "2"}
puts $fileId "$block_transform_xx $block_transform_xy $block_transform_yx $block_transform_yy"
#
#  Field
#
if {$field_plot == "yes"} {
  puts $fileId "1"
  if {$field_format == "pfb"} {puts $fileId "1"}
  if {$field_format == "hdf"} {puts $fileId "2"}
  if {$field_format == "ascii"} {puts $fileId "3"} 
  if {$field_format == "hdf_int1"} {puts $fileId "4"} 
  if {$field_format == "bgr_int1"} {puts $fileId "5"} 
  if {$field_format == "nuft"} {puts $fileId "6"} 
  if {$field_format == "nuft"} {puts $fileId "$field_nuft_timestep"} 
  if {$field_format == "cnb"} {puts $fileId "7"} 
  puts $fileId "$field_file"
  puts $fileId "$field_n_bit $field_color"
  if {$field_data_type == "continuous"} {set field_n_cutoffs 0}
  puts $fileId "$field_n_cutoffs"
  if {$field_n_cutoffs == "0"} {
    puts $fileId "$field_value_min $field_value_max"
    puts $fileId "$field_color_table"
  } else {
    for {set i_cut 1} {$i_cut <= $field_n_cutoffs} {incr i_cut 1} {
      puts $fileId "$field_cutoff($i_cut) \
                    [winfo rgb . $field_cutoff_color($i_cut)]"
    }
  }
  puts $fileId "$field_shade_xy $field_shade_xz $field_shade_yz"
  puts $fileId "$field_corner_bright"
  if {$field_log10 == "yes"} {
    puts $fileId "1"
  } else {
    puts $fileId "0"
  }
#
#  Field legend
#
  set v1 0
  set v2 0
  set v3 0
  if {$field_legend_vertical == "yes"} {set v2 1}
  if {$field_legend_log == "yes"} {set v3 1}
  if {$field_legend == "yes"} {
    if {$field_n_cutoffs > 0} {
      set v1 2
    } elseif {$field_n_cutoffs == 0} {
      set v1 1
    }
  } elseif {$field_legend == "continuous"} {
    set v1 1
  } elseif {$field_legend == "categorical"} {
    set v1 2
  }
  puts $fileId "$v1"
  if {$v1 == 1} {
    puts $fileId "$field_legend_translation_x $field_legend_translation_y"
    puts $fileId "$v2 $v3"
    puts $fileId "$field_legend_min $field_legend_max"
    puts $fileId "$field_legend_height $field_legend_width"
    puts $fileId "$field_legend_label_spacing $field_legend_label_decimals $field_legend_label_tics_per"
    puts $fileId "$field_legend_title"
  } elseif {$v1 == 2} {
    puts $fileId "$field_legend_translation_x $field_legend_translation_y"
    puts $fileId "$field_legend_n_categories $field_legend_n_rows $field_legend_n_columns"
    puts $fileId "$field_legend_height $field_legend_width"
    for {set i 1} {$i <= $field_legend_n_categories} {incr i 1} {
      puts $fileId "$field_legend_category($i)"
      puts $fileId "$field_legend_category_name($i)"
    }
  }
} else {
  puts $fileId "0"
  if {$field_format == "pfb"} {puts $fileId "1"}
  if {$field_format == "hdf"} {puts $fileId "2"}
  if {$field_format == "ascii"} {puts $fileId "3"} 
  if {$field_format == "hdf_int1"} {puts $fileId "4"} 
  if {$field_format == "bgr_int1"} {puts $fileId "5"} 
  if {$field_format == "nuft"} {puts $fileId "6"} 
  if {$field_format == "nuft"} {puts $fileId "$field_nuft_timestep"} 
  puts $fileId "$field_file"
}
#
# Surface
#
puts $fileId "$surface_n_files"
if {$surface_n_files > 0} {
puts $fileId "$surface_z_bottom $surface_z_top"
for {set i 1} {$i <= $surface_n_files} {incr i 1} {
  if {$surface_format($i) == "pfb"} {puts $fileId "1"}
  if {$surface_format($i) == "hdf"} {puts $fileId "2"}
  if {$surface_format($i) == "ascii"} {puts $fileId "3"} 
  if {$surface_format($i) == "hdf_int1"} {puts $fileId "4"} 
  if {$surface_format($i) == "bgr_int1"} {puts $fileId "5"} 
  if {$surface_format($i) == "nuft"} {puts $fileId "6"} 
  if {$surface_format($i) == "nuft"} {puts $fileId "$surface_nuft_timestep($i)"} 
  if {$surface_format($i) == "cnb"} {puts $fileId "7"} 
  puts $fileId "$surface_file($i)" 
  set n_cut $surface_n_cutoffs($i) 
  for {set j 1} {$j <= $surface_n_cutoffs($i)} {incr j 1} {
    if {$surface_plot($i,$j) == "no"} { set n_cut [expr $n_cut - 1]}
  }
  puts $fileId "$n_cut"
  for {set j 1} {$j <= $surface_n_cutoffs($i)} {incr j 1} {
   if {$surface_plot($i,$j) == "no"} {
    } else {
    set surface_transparent($i,$j,out) [expr 1.0 - $transparency($i,$j)]
    set surface_transparent($i,$j,in) [expr 1.0 - $transparency($i,$j)]
    set surface_transparent($i,$j,edge) [expr 1.0 - $transparency($i,$j)]
    puts $fileId "$surface_value_min($i,$j) $surface_value_max($i,$j)"
    puts $fileId "$surface_plot($i,$j,out) $surface_plot($i,$j,in) $surface_plot($i,$j,edge)"
    puts $fileId "$surface_transparent($i,$j,out) $surface_transparent($i,$j,in) $surface_transparent($i,$j,edge)"
    puts $fileId "$surface_transparent_dia($i,$j)"
  foreach k {out in edge} {
    set rgb "[winfo rgb . $surface_color($i,$j,$k)]"
    set r [lindex $rgb 0]
    set g [lindex $rgb 1]
    set b [lindex $rgb 2]
    set r [expr $r/65535.]
    set g [expr $g/65535.]
    set b [expr $b/65535.]
    set v1 [ expr $r + ($surface_shade_xy($i,$j) + $surface_shade_bottom)/15. ]
    set v2 [ expr $g + ($surface_shade_xy($i,$j) + $surface_shade_bottom)/15. ]
    set v3 [ expr $b + ($surface_shade_xy($i,$j) + $surface_shade_bottom)/15. ]
    set v4 [ expr $r + ($surface_shade_xz($i,$j) + $surface_shade_bottom)/15. ]
    set v5 [ expr $g + ($surface_shade_xz($i,$j) + $surface_shade_bottom)/15. ]
    set v6 [ expr $b + ($surface_shade_xz($i,$j) + $surface_shade_bottom)/15. ]
    set v7 [ expr $r + ($surface_shade_yz($i,$j) + $surface_shade_bottom)/15. ]
    set v8 [ expr $g + ($surface_shade_yz($i,$j) + $surface_shade_bottom)/15. ]
    set v9 [ expr $b + ($surface_shade_yz($i,$j) + $surface_shade_bottom)/15. ]
    if {$v1 < 0.0} {set v1 0.0}
    if {$v1 > 1.0} {set v1 1.0}
    if {$v2 < 0.0} {set v2 0.0}
    if {$v2 > 1.0} {set v2 1.0}
    if {$v3 < 0.0} {set v3 0.0}
    if {$v3 > 1.0} {set v3 1.0}
    if {$v4 < 0.0} {set v4 0.0}
    if {$v4 > 1.0} {set v4 1.0}
    if {$v5 < 0.0} {set v5 0.0}
    if {$v5 > 1.0} {set v5 1.0}
    if {$v6 < 0.0} {set v6 0.0}
    if {$v6 > 1.0} {set v6 1.0}
    if {$v7 < 0.0} {set v7 0.0}
    if {$v7 > 1.0} {set v7 1.0}
    if {$v8 < 0.0} {set v8 0.0}
    if {$v8 > 1.0} {set v8 1.0}
    if {$v9 < 0.0} {set v9 0.0}
    if {$v9 > 1.0} {set v9 1.0}
puts $fileId [format "%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f" \
$v1 $v2 $v3  $v4 $v5 $v6  $v7 $v8 $v9 ]
    set v1 [expr $r + ($surface_shade_xy($i,$j) + $surface_shade_top)/15. ]
    set v2 [expr $g + ($surface_shade_xy($i,$j) + $surface_shade_top)/15. ]
    set v3 [expr $b + ($surface_shade_xy($i,$j) + $surface_shade_top)/15. ]
    set v4 [expr $r + ($surface_shade_xz($i,$j) + $surface_shade_top)/15. ]
    set v5 [expr $g + ($surface_shade_xz($i,$j) + $surface_shade_top)/15. ]
    set v6 [expr $b + ($surface_shade_xz($i,$j) + $surface_shade_top)/15. ]
    set v7 [expr $r + ($surface_shade_yz($i,$j) + $surface_shade_top)/15. ]
    set v8 [expr $g + ($surface_shade_yz($i,$j) + $surface_shade_top)/15. ]
    set v9 [expr $b + ($surface_shade_yz($i,$j) + $surface_shade_top)/15. ]
    if {$v1 < 0.0} {set v1 0.0}
    if {$v1 > 1.0} {set v1 1.0}
    if {$v2 < 0.0} {set v2 0.0}
    if {$v2 > 1.0} {set v2 1.0}
    if {$v3 < 0.0} {set v3 0.0}
    if {$v3 > 1.0} {set v3 1.0}
    if {$v4 < 0.0} {set v4 0.0}
    if {$v4 > 1.0} {set v4 1.0}
    if {$v5 < 0.0} {set v5 0.0}
    if {$v5 > 1.0} {set v5 1.0}
    if {$v6 < 0.0} {set v6 0.0}
    if {$v6 > 1.0} {set v6 1.0}
    if {$v7 < 0.0} {set v7 0.0}
    if {$v7 > 1.0} {set v7 1.0}
    if {$v8 < 0.0} {set v8 0.0}
    if {$v8 > 1.0} {set v8 1.0}
    if {$v9 < 0.0} {set v9 0.0}
    if {$v9 > 1.0} {set v9 1.0}
puts $fileId [format "%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f" \
$v1 $v2 $v3  $v4 $v5 $v6  $v7 $v8 $v9 ]
    }
   }
  }
#
# Surface legend
#
  if {$surface_legend($i) == "yes"} {
    puts $fileId "1"
    puts $fileId "$surface_legend_translation_x($i) $surface_legend_translation_y($i)" 
    for {set j 1} {$j <= $surface_n_cutoffs($i)} {incr j 1} {
      if {$surface_plot($i,$j) == "no" } {  
      } else {
      puts $fileId "$surface_legend_label($i,$j)"
      }
    }
  } else {
    puts $fileId "0"
  }
 }
}
#
#  Lines
#
puts $fileId "$line_n_files"
for {set i 1} {$i <= $line_n_files} {incr i 1} {
  if {$line_format($i) == "slim"} {puts $fileId "1"} 
  if {$line_format($i) == "nts"} {puts $fileId "2"} 
  if {$line_format($i) == "nuft"} {puts $fileId "3"} 
  if {$line_format($i) == "ind"} {puts $fileId "4"} 
  puts $fileId "$line_file($i)" 
  puts $fileId "$line_value_min($i) $line_value_max($i)"
  if {$line_value_min($i) == $line_value_max($i)} {
    puts $fileId "[winfo rgb . $line_color($i)]"
  } else {
    puts $fileId "$line_color_table($i)"
  }
  puts $fileId "$line_origin_x($i) $line_origin_y($i) $line_origin_z($i)"
  puts $fileId "$line_winnow($i) $line_time_scale($i)"
  set marker [expr $line_outside_marker($i) * $line_outside_sign($i) ] 
  puts $fileId "$line_outside_linewidth($i) $line_outside_dash($i) $marker $line_outside_linewidth_min($i)"
  set marker [expr $line_inside_marker($i) * $line_inside_sign($i) ] 
  puts $fileId "$line_inside_linewidth($i) $line_inside_dash($i) $marker $line_inside_linewidth_min($i)"
#
#   Line legend 
#
  if {$line_legend($i) == "yes"} {
    puts $fileId "1"
    puts $fileId "$line_legend_translation_x($i) $line_legend_translation_y($i)" 
    puts $fileId "$line_legend_height($i) $line_legend_width($i)"
    puts $fileId "$line_legend_title($i)"
    puts $fileId "$line_legend_label_spacing($i) $line_legend_label_decimals($i) $line_legend_label_tics_per($i)"
  } else {
    puts $fileId "0"
  }
}
#
# Dots
#
puts $fileId "$dots_n_files"
for {set i 1} {$i <= $dots_n_files} {incr i 1} {
  puts $fileId "$dots_file($i)" 
  puts $fileId "$dots_value_min($i) $dots_value_max($i)"
  if {$dots_value_min($i) == $dots_value_max($i)} {
    puts $fileId "[winfo rgb . $dots_color($i)]"
  } else {
    puts $fileId "$dots_color_table($i)"
  }
  puts $fileId "$dots_origin_x($i) $dots_origin_y($i) $dots_origin_z($i)"
  puts $fileId "$dots_winnow($i) $dots_time_scale($i)"
  puts $fileId "$dots_outside_size($i) $dots_outside_marker($i) $dots_outside_size_min($i)"
  puts $fileId "$dots_inside_size($i) $dots_inside_marker($i) $dots_inside_size_min($i)"
#
#   Dots legend 
#
  if {$dots_legend($i) == "yes"} {
    puts $fileId "1"
    puts $fileId "$dots_legend_translation_x($i) $dots_legend_translation_y($i)" 
    puts $fileId "$dots_legend_height($i) $dots_legend_width($i)"
    puts $fileId "$dots_legend_title($i)"
    puts $fileId "$dots_legend_label_spacing($i) $dots_legend_label_decimals($i) $dots_legend_label_tics_per($i)"
  } else {
    puts $fileId "0"
  }
}
close $fileId
#
#  Run Chunk
#
set fileId [open "dummy_file.par" w 0600] 
puts $fileId "$par_file"
close $fileId
exec $exe_file < dummy_file.par > chunk.out
file delete -force dummy_file.par 
#file delete -force chunk.out 
