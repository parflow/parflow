#
# Chunk file locations
#
set exe_file "$home_dir/code/chunk.o"
set par_file "$home_dir/code/chunk.par" 
set tcl_file "$home_dir/tcl/chunk_input.tcl"
set chunk_run_file "$home_dir/src/chunk_run_0701.tcl"
set ps_file "chunk.ps"
set global_file "$home_dir/src/chunk_global.tcl" 
set color_table_dir "$home_dir/ctb"
#
# Input file defaults
#
set field_file_default "$home_dir/test/field_test.ascii"
set surface_file_default "$home_dir/test/field_test.ascii"
set line_file_default "$home_dir/test/wells.lin"
set dots_file_default "$home_dir/test/dots.txt"
set color_table_default "$home_dir/ctb/rainbow10.ctb" 
#
# EPS parameters
#
set ps_file_format "ps"
set eps_BoundingBox_min_x 0.
set eps_BoundingBox_min_y 0.
set eps_BoundingBox_max_x 612.
set eps_BoundingBox_max_y 792.
set eps_creator "Chunk Version 4.0"
set eps_date "[clock format [clock seconds] -format %x ]"
set eps_time "[clock format [clock seconds] -format %X ]" 
#
# Plot
#
set plot_translation_x 0.0
set plot_translation_y 0.0
set plot_landscape "no"
set plot_textsize 10.
set plot_background_color "#FFFFFF"
set plot_text_color "#000000"
set plot_axes_color "#000000"
set plot_title_1 " "
set plot_title_2 " "
set plot_title_translation_x 0.0
set plot_title_translation_y 0.0
set plot_title_justification "center"
#
# Block
#
set block_min_x  0.0
set block_min_y  0.0
set block_min_z  0.0 
set block_d_x 1.0
set block_d_y 1.0
set block_d_z 1.0
set block_scale_x 5.
set block_scale_y 5.
set block_scale_z  5.
set block_transform_xx 0.93
set block_transform_xy -0.4
set block_transform_yx 0.93
set block_transform_yy 0.4
#
# Labeling
#
set label_x "x (m)"
set label_y "y (m)"
set label_z "z (m)"
set label_space_x 5.
set label_space_y 5.
set label_space_z 5.
set label_decimals_x 0
set label_decimals_y 0
set label_decimals_z 0
set label_tic_per_x 5  
set label_tic_per_y 5 
set label_tic_per_z 5 
#
# Chunk
#
set chunk_n_x 2
set chunk_n_y 2
set chunk_n_z 2
set chunk_space_x 0.
set chunk_space_y 0.
set chunk_space_z 0.
set chunk_crop_x_min 0
set chunk_crop_x_max 0
set chunk_crop_y_min 0
set chunk_crop_y_max 0
set chunk_crop_z_min 0
set chunk_crop_z_max 0
set chunk_erase_n 1
set chunk_erase_list "8" 
set chunk_view "3D"

for {set k 1} {$k <= 20} {incr k 1} {
for {set j 1} {$j <= 20} {incr j 1} {
for {set i 1} {$i <= 20} {incr i 1} {
set chunk_num($i,$j,$k) 0
} } }
set chunk_num(2,2,2) 1
#
# Field
#
set field_data_type "continuous"
set field_plot "yes"
set field_format "ascii"
set field_file $field_file_default 
set field_nuft_timestep 1
set field_value_min 0.0
set field_value_max 5.0
set field_color_table $color_table_default 
set field_color 1
set field_n_bit 4
set field_shade_xy 2
set field_shade_xz 0
set field_shade_yz -2
set field_corner_bright 0.8
set field_log10 "no"
#
# Field Categorical
#
set field_n_cutoffs 1
foreach i {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20} {
set field_cutoff($i) [expr $i * 1.0] 
}
set field_cutoff_color(1) "#000000"
set field_cutoff_color(2) "#0000ff"
set field_cutoff_color(3) "#00ff00"
set field_cutoff_color(4) "#ffff00"
set field_cutoff_color(5) "#ff0000"
set field_cutoff_color(6) "#ffffff"
set field_cutoff_color(7) $field_cutoff_color(1)
set field_cutoff_color(8) $field_cutoff_color(2)
set field_cutoff_color(9) $field_cutoff_color(3)
set field_cutoff_color(10) $field_cutoff_color(4)
set field_cutoff_color(11) $field_cutoff_color(5)
set field_cutoff_color(12) $field_cutoff_color(6)
set field_cutoff_color(13) $field_cutoff_color(1)
set field_cutoff_color(14) $field_cutoff_color(2)
set field_cutoff_color(15) $field_cutoff_color(3)
set field_cutoff_color(16) $field_cutoff_color(4)
set field_cutoff_color(17) $field_cutoff_color(5)
set field_cutoff_color(18) $field_cutoff_color(6)
set field_cutoff_color(19) $field_cutoff_color(1)
set field_cutoff_color(20) $field_cutoff_color(2)
#
# Field legend
#
set field_legend "no" 
set field_legend_log "no"
set field_legend_vertical "no"
set field_legend_title " "
set field_legend_translation_x 72.00 
set field_legend_translation_y 72.00 
#
#   Continuous
#
set field_legend_log "no"
set field_legend_vertical "no"
set field_legend_min  0.
set field_legend_max  5.
set field_legend_height 0.25
set field_legend_width 3.0
set field_legend_label_spacing 1.0
set field_legend_label_decimals 0
set field_legend_label_tics_per 5 
#
#   Categorical
#
set field_legend_n_categories $field_n_cutoffs 
set field_legend_n_rows 3
set field_legend_n_columns 2
for {set i 1} {$i <= 20} {incr i 1} {
set field_legend_category($i) $i 
set field_legend_category_name($i) "Category $i"
}
#
# Surface
#
set surface_n_files 0
set surface_z_bottom 1
set surface_z_top 1
set surface_shade_bottom  0.0
set surface_shade_top 0.0
foreach j {1 2 3 4 5 6 7 8 9 10} {
#
# Surface #1
#
set surface_format($j) "ascii"
set surface_file($j) $surface_file_default
set surface_n_cutoffs($j) 1
set surface_nuft_timestep($j) 1
#
# Surface Cutoff #1
#
foreach i {1 2 3 4 5 6 7 8 9 10} { 
set surface_plot($j,$i) "yes"
set surface_value_min($j,$i) 3.0
set surface_value_max($j,$i) 6.0
foreach k {out in edge} {set surface_color($j,$i,$k) "gray"} 
set surface_shade_xy($j,$i)  -2 
set surface_shade_xz($j,$i)  0 
set surface_shade_yz($j,$i)   2 
set surface_plot($j,$i,out)   -1
set surface_plot($j,$i,in)   -1
set surface_plot($j,$i,edge)  0
set transparency($j,$i) 0.0
set surface_transparent($j,$i,out)  1.0
set surface_transparent($j,$i,in)   1.0
set surface_transparent($j,$i,edge) 1.0
set surface_transparent_dia($j,$i) 3
set surface_legend_label($j,$i) "> $i"
}
#
# Surface Legend
#
set surface_legend($j) "yes"
set surface_legend_translation_x($j) 36.0 
set surface_legend_translation_y($j) 36.0 
}
#
#
# Lines
#
set line_n_files 0
#
# Line 1
#
foreach i {1 2 3 4 5 6 7 8 9 10} {
set line_format($i) "nuft"
set line_file($i) $line_file_default 
set line_value_min($i) 0.
set line_value_max($i) 0.
set line_color($i) "gray"
set line_color_table($i) $color_table_default
set line_origin_x($i) 0.0
set line_origin_y($i) 0.0
set line_origin_z($i) 0.0
set line_winnow($i) 1
set line_time_scale($i) 1.00
set line_outside_linewidth($i) 2.0
set line_outside_linewidth_min($i) 1.0
set line_outside_dash($i) 0
set line_outside_marker($i) 0 
set line_outside_sign($i) 1 
set line_inside_linewidth($i) 2.0
set line_inside_linewidth_min($i) 1.0
set line_inside_dash($i) 2
set line_inside_marker($i) 0 
set line_inside_sign($i) 1 
#
#  Line 1 legend
#
set line_legend($i) "no"
set line_legend_translation_x($i) 108.00 
set line_legend_translation_y($i) 108.00 
set line_legend_height($i) 0.25
set line_legend_width($i) 2.5
set line_legend_title($i) "Line $i"
set line_legend_label_spacing($i) 10.0
set line_legend_label_decimals($i) 0
set line_legend_label_tics_per($i) 5
}
#
# Dots
#
set dots_n_files 0
#
# Dots 1
#
foreach i {1 2 3 4 5 6 7 8 9 10} {
set dots_file($i) $dots_file_default 
set dots_value_min($i) 0.0
set dots_value_max($i) 0.0 
set dots_time_scale($i) 1.00
set dots_color_table($i) $color_table_default
set dots_origin_x($i) 0.0
set dots_origin_y($i) 0.0
set dots_origin_z($i) 0.0
set dots_winnow($i) 1 
set dots_outside_size($i) 5.0
set dots_outside_size_min($i) 1.0
set dots_outside_marker($i) 0
set dots_inside_size($i) 5.0
set dots_inside_size_min($i) 1.0
set dots_inside_marker($i) 0
set dots_color($i) "gray" 
set dots_format($i) "slim"
#
#  Dots legend 1
#
set dots_legend($i) "no"
set dots_legend_translation_x($i) 144.0 
set dots_legend_translation_y($i) 144.0 
set dots_legend_height($i) 0.25
set dots_legend_width($i) 2.50
set dots_legend_title($i) "Age (yr)"
set dots_legend_label_spacing($i) 1. 
set dots_legend_label_decimals($i) 0
set dots_legend_label_tics_per($i) 10
}
