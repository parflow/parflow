set fileId [open $tcl_file w 0600]

#
# Chunk file locations
#
puts $fileId "#"
puts $fileId "# Files"
puts $fileId "#"
puts $fileId "set home_dir         {$home_dir}"
puts $fileId "set exe_file         {$exe_file}"
puts $fileId "set par_file         {$par_file}" 
puts $fileId "set tcl_file         {$tcl_file}"
puts $fileId "set chunk_run_file   {$chunk_run_file}" 
puts $fileId "set ps_file          {$ps_file}"
puts $fileId "set global_file      {$global_file}" 
#
# EPS parameters
#
puts $fileId "#"
puts $fileId "# PS/EPS Parameters"
puts $fileId "#"
puts $fileId "set ps_file_format $ps_file_format"
puts $fileId "set eps_BoundingBox_min_x $eps_BoundingBox_min_x"
puts $fileId "set eps_BoundingBox_min_y $eps_BoundingBox_min_y"
puts $fileId "set eps_BoundingBox_max_x $eps_BoundingBox_max_x"
puts $fileId "set eps_BoundingBox_max_y $eps_BoundingBox_max_y"
puts $fileId "set eps_creator {$eps_creator}"
puts $fileId "set eps_date {$eps_date}"
puts $fileId "set eps_time {$eps_time}"
#
# Plot
#
puts $fileId "#"
puts $fileId "# Plot"
puts $fileId "#"
puts $fileId "set plot_translation_x $plot_translation_x"
puts $fileId "set plot_translation_y $plot_translation_y"
puts $fileId "set plot_landscape $plot_landscape"
puts $fileId "set plot_textsize $plot_textsize"
puts $fileId "set plot_background_color $plot_background_color"
puts $fileId "set plot_text_color $plot_text_color"
puts $fileId "set plot_axes_color $plot_axes_color"
puts $fileId "set plot_title_1 {$plot_title_1}" 
puts $fileId "set plot_title_2 {$plot_title_2}" 
puts $fileId "set plot_title_translation_x $plot_title_translation_x"
puts $fileId "set plot_title_translation_y $plot_title_translation_y"
puts $fileId "set plot_title_justification $plot_title_justification"
#
# Block
#
puts $fileId "#"
puts $fileId "# Block"
puts $fileId "#"
puts $fileId "set block_min_x $block_min_x"
puts $fileId "set block_min_y $block_min_y"
puts $fileId "set block_min_z $block_min_z"
puts $fileId "set block_d_x $block_d_x"
puts $fileId "set block_d_y $block_d_y"
puts $fileId "set block_d_z $block_d_z"
puts $fileId "set block_scale_x $block_scale_x"
puts $fileId "set block_scale_y $block_scale_y"
puts $fileId "set block_scale_z $block_scale_z"
puts $fileId "set block_transform_xx $block_transform_xx"
puts $fileId "set block_transform_xy $block_transform_xy"
puts $fileId "set block_transform_yx $block_transform_yx"
puts $fileId "set block_transform_yy $block_transform_yy"
puts $fileId "set label_x {$label_x}"
puts $fileId "set label_y {$label_y}"
puts $fileId "set label_z {$label_z}"
puts $fileId "set label_space_x $label_space_x"
puts $fileId "set label_space_y $label_space_y"
puts $fileId "set label_space_z $label_space_z"
puts $fileId "set label_decimals_x $label_decimals_x"
puts $fileId "set label_decimals_y $label_decimals_y"
puts $fileId "set label_decimals_z $label_decimals_z"
puts $fileId "set label_tic_per_x $label_tic_per_x"  
puts $fileId "set label_tic_per_y $label_tic_per_y"  
puts $fileId "set label_tic_per_z $label_tic_per_z"  
#
# Chunk
#
puts $fileId "#"
puts $fileId "# Chunk"
puts $fileId "#"
puts $fileId "set chunk_n_x $chunk_n_x"
puts $fileId "set chunk_n_y $chunk_n_y"
puts $fileId "set chunk_n_z $chunk_n_z"
puts $fileId "set chunk_space_x $chunk_space_x"
puts $fileId "set chunk_space_y $chunk_space_y"
puts $fileId "set chunk_space_z $chunk_space_z"
puts $fileId "set chunk_crop_x_min $chunk_crop_x_min"
puts $fileId "set chunk_crop_x_max $chunk_crop_x_max"
puts $fileId "set chunk_crop_y_min $chunk_crop_y_min"
puts $fileId "set chunk_crop_y_max $chunk_crop_y_max"
puts $fileId "set chunk_crop_z_min $chunk_crop_z_min"
puts $fileId "set chunk_crop_z_max $chunk_crop_z_max"
puts $fileId "set chunk_erase_n $chunk_erase_n"
puts $fileId "set chunk_erase_list {$chunk_erase_list}" 
puts $fileId "set chunk_view $chunk_view"
for {set k 1} {$k <= $chunk_n_z} {incr k 1} {
for {set j 1} {$j <= $chunk_n_y} {incr j 1} {
for {set i 1} {$i <= $chunk_n_x} {incr i 1} {
puts $fileId "set chunk_num($i,$j,$k) $chunk_num($i,$j,$k)"
}}}
#
# Field
#
puts $fileId "#"
puts $fileId "# Field"
puts $fileId "#"
puts $fileId "set field_data_type $field_data_type"
puts $fileId "set field_plot $field_plot"
puts $fileId "set field_format $field_format"
puts $fileId "set field_file {$field_file}"
puts $fileId "set field_nuft_timestep $field_nuft_timestep"
puts $fileId "set field_n_cutoffs $field_n_cutoffs"
for {set i 1} {$i <= $field_n_cutoffs} {incr i 1} {
  puts $fileId "set field_cutoff($i) $field_cutoff($i)"
  puts $fileId "set field_cutoff_color($i) $field_cutoff_color($i)"
}
puts $fileId "set field_value_min $field_value_min"
puts $fileId "set field_value_max $field_value_max"
puts $fileId "set field_color_table {$field_color_table}"
puts $fileId "set field_color $field_color"
puts $fileId "set field_n_bit $field_n_bit"
puts $fileId "set field_shade_xy $field_shade_xy"
puts $fileId "set field_shade_xz $field_shade_xz"
puts $fileId "set field_shade_yz $field_shade_yz"
puts $fileId "set field_corner_bright $field_corner_bright"
puts $fileId "set field_log10 $field_log10"
#
# Field legend
#
puts $fileId "#"
puts $fileId "# Field Legend"
puts $fileId "#"
puts $fileId "set field_legend $field_legend" 
puts $fileId "set field_legend_log $field_legend_log"
puts $fileId "set field_legend_vertical $field_legend_vertical"
puts $fileId "set field_legend_title {$field_legend_title}"
puts $fileId "set field_legend_translation_x $field_legend_translation_x"
puts $fileId "set field_legend_translation_y $field_legend_translation_y"
#
#   Continuous
#
puts $fileId "#"
puts $fileId "# Field Legend - Continuous"
puts $fileId "#"
puts $fileId "set field_legend_log $field_legend_log"
puts $fileId "set field_legend_vertical $field_legend_vertical"
puts $fileId "set field_legend_min $field_legend_min"
puts $fileId "set field_legend_max $field_legend_max"
puts $fileId "set field_legend_height $field_legend_height"
puts $fileId "set field_legend_width $field_legend_width"
puts $fileId "set field_legend_label_spacing $field_legend_label_spacing"
puts $fileId "set field_legend_label_decimals $field_legend_label_decimals"
puts $fileId "set field_legend_label_tics_per $field_legend_label_tics_per"
#
#   Categorical
#
puts $fileId "#"
puts $fileId "# Field Legend - Categorical"
puts $fileId "#"
puts $fileId "set field_legend_n_categories $field_legend_n_categories"
puts $fileId "set field_legend_n_rows $field_legend_n_rows"
puts $fileId "set field_legend_n_columns $field_legend_n_columns"
for {set c 1} {$c <= $field_legend_n_categories} {incr c 1} {  
  puts $fileId "set field_legend_category($c) $field_legend_category($c)"
  puts $fileId "set field_legend_category_name($c) {$field_legend_category_name($c)}"
}
#
# Surface
#
puts $fileId "#"
puts $fileId "# Surface"
puts $fileId "#"
puts $fileId "set surface_n_files $surface_n_files"
puts $fileId "set surface_z_bottom  $surface_z_bottom"
puts $fileId "set surface_z_top $surface_z_top"
puts $fileId "set surface_shade_bottom $surface_shade_bottom"
puts $fileId "set surface_shade_top $surface_shade_top"
for {set j 1} {$j <= $surface_n_files} {incr j 1} {
#
# Surface #1
#
puts $fileId "#"
puts $fileId "#   Surface #j"
puts $fileId "#"
puts $fileId "set surface_format($j) $surface_format($j)"
puts $fileId "set surface_file($j) {$surface_file($j)}"
puts $fileId "set surface_n_cutoffs($j) $surface_n_cutoffs($j)"
puts $fileId "set surface_nuft_timestep($j) $surface_nuft_timestep($j)"
#
# Surface Cutoff #1
#
for {set k 1} {$k <= $surface_n_cutoffs($j)} {incr k 1} {
puts $fileId "#"
puts $fileId "#      Surface Cutoff #$k"
puts $fileId "#"
puts $fileId "set surface_plot($j,$k) $surface_plot($j,$k)"
puts $fileId "set surface_value_min($j,$k) $surface_value_min($j,$k)"
puts $fileId "set surface_value_max($j,$k) $surface_value_max($j,$k)"
foreach l {out in edge} {
set surface_transparent($j,$k,$l) [expr 1.0 - $transparency($j,$k)]
puts $fileId "set surface_color($j,$k,$l) $surface_color($j,$k,$l)"
puts $fileId "set surface_plot($j,$k,$l) $surface_plot($j,$k,$l)"
puts $fileId "set surface_transparent($j,$k,$l) $surface_transparent($j,$k,$l)"
puts $fileId "set surface_transparenti_dia($j,$k) $surface_transparent_dia($j,$k)"
}
puts $fileId "set surface_shade_xy($j,$k) $surface_shade_xy($j,$k)"
puts $fileId "set surface_shade_xz($j,$k) $surface_shade_xz($j,$k)"
puts $fileId "set surface_shade_yz($j,$k) $surface_shade_yz($j,$k)"
puts $fileId "set surface_legend_label($j,$k) {$surface_legend_label($j,$k)}"

}
#
# Surface Legend
#
puts $fileId "#"
puts $fileId "# Surface Legend"
puts $fileId "#"
puts $fileId "set surface_legend($j) $surface_legend($j)"
puts $fileId "set surface_legend_translation_x($j) $surface_legend_translation_x($j)" 
puts $fileId "set surface_legend_translation_y($j) $surface_legend_translation_y($j)" 
}
#
#
# Lines
#
puts $fileId "#"
puts $fileId "# Lines"
puts $fileId "#"
puts $fileId "set line_n_files $line_n_files"
for {set k 1} {$k <= $line_n_files} {incr k 1} {
puts $fileId "#"
puts $fileId "#   Line $k"
puts $fileId "#"
puts $fileId "set line_format($k) $line_format($k)"
puts $fileId "set line_file($k) {$line_file($k)}" 
puts $fileId "set line_value_min($k) $line_value_min($k)"
puts $fileId "set line_value_max($k) $line_value_max($k)"
puts $fileId "set line_color($k) $line_color($k)"
puts $fileId "set line_color_table($k) {$line_color_table($k)}"
puts $fileId "set line_origin_x($k) $line_origin_x($k)"
puts $fileId "set line_origin_y($k) $line_origin_y($k)"
puts $fileId "set line_origin_z($k) $line_origin_z($k)"
puts $fileId "set line_winnow($k) $line_winnow(1)"
puts $fileId "set line_time_scale($k) $line_time_scale($k)"
puts $fileId "set line_outside_linewidth($k) $line_outside_linewidth($k)"
puts $fileId "set line_outside_linewidth_min($k) $line_outside_linewidth_min($k)"
puts $fileId "set line_outside_dash($k) $line_outside_dash($k)"
puts $fileId "set line_outside_marker($k) $line_outside_marker($k)"
puts $fileId "set line_outside_sign($k) $line_outside_sign($k)"
puts $fileId "set line_inside_linewidth($k) $line_inside_linewidth($k)"
puts $fileId "set line_inside_linewidth_min($k) $line_inside_linewidth_min($k)"
puts $fileId "set line_inside_dash($k) $line_inside_dash($k)"
puts $fileId "set line_inside_marker($k) $line_inside_marker($k)"
puts $fileId "set line_inside_sign($k) $line_inside_sign($k)"
puts $fileId "#"
puts $fileId "#   Line $k Legend"
puts $fileId "#"
puts $fileId "set line_legend($k) $line_legend($k)"
puts $fileId "set line_legend_translation_x($k) $line_legend_translation_x($k)"
puts $fileId "set line_legend_translation_y($k) $line_legend_translation_y($k)"
puts $fileId "set line_legend_height($k) $line_legend_height($k)"
puts $fileId "set line_legend_width($k) $line_legend_width($k)"
puts $fileId "set line_legend_title($k) {$line_legend_title($k)}"
puts $fileId "set line_legend_label_spacing($k) $line_legend_label_spacing($k)"
puts $fileId "set line_legend_label_decimals($k) $line_legend_label_spacing($k)" 
puts $fileId "set line_legend_label_tics_per($k) $line_legend_label_tics_per($k)" 
}
#
# Dots
#
puts $fileId "#"
puts $fileId "# Dots" 
puts $fileId "#"
puts $fileId "set dots_n_files $dots_n_files"
for {set k 1} {$k <= $dots_n_files} {incr k 1} {
puts $fileId "#"
puts $fileId "#   Dots $k"
puts $fileId "#"
puts $fileId "set dots_file($k) {$dots_file($k)}" 
puts $fileId "set dots_value_min($k) $dots_value_min($k)"
puts $fileId "set dots_value_max($k) $dots_value_max($k)"
puts $fileId "set dots_color($k) $dots_color($k)"
puts $fileId "set dots_color_table($k) {$dots_color_table($k)}"
puts $fileId "set dots_origin_x($k) $dots_origin_x($k)"
puts $fileId "set dots_origin_y($k) $dots_origin_y($k)"
puts $fileId "set dots_origin_z($k) $dots_origin_z($k)"
puts $fileId "set dots_winnow($k) $dots_winnow($k)"
puts $fileId "set dots_time_scale($k) $dots_time_scale($k)"
puts $fileId "set dots_outside_size($k) $dots_outside_size($k)"
puts $fileId "set dots_outside_size_min($k) $dots_outside_size_min($k)"
puts $fileId "set dots_outside_marker($k) $dots_outside_marker($k)"
puts $fileId "set dots_inside_size($k) $dots_inside_size($k)"
puts $fileId "set dots_inside_size_min($k) $dots_inside_size_min($k)"
puts $fileId "set dots_inside_marker($k) $dots_inside_marker($k)"
puts $fileId "#"
puts $fileId "#   Dots $k Legend"
puts $fileId "#"
puts $fileId "set dots_legend($k) $dots_legend($k)"
puts $fileId "set dots_legend_translation_x($k) $dots_legend_translation_x($k)"
puts $fileId "set dots_legend_translation_y($k) $dots_legend_translation_y($k)"
puts $fileId "set dots_legend_height($k) $dots_legend_height($k)"
puts $fileId "set dots_legend_width($k) $dots_legend_width($k)"
puts $fileId "set dots_legend_title($k) {$dots_legend_title($k)}"
puts $fileId "set dots_legend_label_spacing($k) $dots_legend_label_spacing($k)"
puts $fileId "set dots_legend_label_decimals($k) $dots_legend_label_spacing($k)" 
puts $fileId "set dots_legend_label_tics_per($k) $dots_legend_label_tics_per($k)" 
}
close $fileId
