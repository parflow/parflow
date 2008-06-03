APPS.SingleWindowApp Orthograph<NExOffset=23.,NEyOffset=-34.,NEscalingFactor=1.> {
   UI {
      shell {
         x = 0;
         y = 146;
      };
      Windows {
         IUI {
            optionList {
               selectedItem = 0;
            };
         };
      };
      Editors {
         IUI {
            optionList {
               selectedItem = 0;
            };
         };
      };
      Modules {
         IUI {
            optionList {
               cmdList => {
                  <-.<-.<-.<-.orthograph_UI.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.scale_grid.UImod_panel.option,
                  <-.<-.<-.<-.orthograph_display.isoline.UIpanel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.downsize.panel.option,
                  <-.<-.<-.<-.orthograph_display.bounds.UIpanel.option,
                  <-.<-.<-.<-.orthograph_display.Axis2D.UIpanel.option,
                  <-.<-.<-.<-.orthograph_display.probe.UIpanel.option
               };
               selectedItem = 0;
            };
            panel {
               visible = 1;
            };
         };
      };
   };
   macro orthograph_display<NEx=297.,NEy=341.,NExOffset=135.,NEyOffset=30.> {
      MODS.isoline isoline<NEx=330.,NEy=143.> {
         IsoParam {
            level_min => <-.<-.min.output;
            level_max => <-.<-.max.output;
            ncontours = 25;
         };
         in_field => <-.Mesh_plot_2D_xform.out;
      };
      USER.Mesh_plot_2D_xform Mesh_plot_2D_xform<NEx=154.,NEy=143.> {
         in => <-.DVorthoslice_unif.out;
         axis => <-.axis;
      };
      DVM.DVorthoslice_unif DVorthoslice_unif<NEx=264.,NEy=44.> {
         in<NEportLevels={3,0}> => <-.<-.PF_Grid_from_File.Hack;
         axis => <-.axis;
         plane => <-.plane;
         out {
            node_data<NEdisplayMode="showParams">;
         };
      };
      MODS.bounds bounds<NEx=154.,NEy=198.> {
         in_field => <-.Mesh_plot_2D_xform.out;
      };
      GDM.Uviewer2D Uviewer2D<NEx=198.,NEy=297.> {
         Scene {
            Top {
               child_objs => {
                  <-.<-.<-.bounds.out_obj,<-.<-.<-.isoline.out_obj,
                  <-.<-.<-.Axis2D.out_obj,<-.<-.<-.probe.out_obj};
            };
         };
      };
      int axis<NEportLevels={2,1},NEx=-66.,NEy=33.> => <-.orthograph_UI.axis_number.axis;
      int plane<NEportLevels={2,1},NEx=-44.,NEy=0.> => <-.orthograph_UI.plane;
      GMOD.copy_on_change min<NEx=-77.,NEy=77.> {
         input => Mesh_plot_2D_xform.in.node_data[0].min;
      };
      GMOD.copy_on_change max<NEx=-77.,NEy=110.> {
         input => Mesh_plot_2D_xform.in.node_data[0].max;
      };
      GEOMS.Axis2D Axis2D<NEx=330.,NEy=198.> {
         in_field => <-.Mesh_plot_2D_xform.out;
         x_axis_param {
            minor_ticks = 0;
            off_anno => 0.2;
            axis_name => switch((2 - (<-.<-.axis == 0)),"Y","X");
         };
         y_axis_param {
            minor_ticks = 0;
            off_anno => 3.;
            axis_name => switch((2 - (<-.<-.axis == 2)),"Y","Z");
         };
         Axis2DUI {
            x_start_typein {
               x = 0;
            };
            x_end_typein {
               x = 0;
            };
            x_origin_typein {
               x = 0;
            };
            x_step_typein {
               x = 0;
            };
            x_offset_typein {
               x = 0;
            };
            y_start_typein {
               x = 0;
            };
            y_end_typein {
               x = 0;
            };
            y_origin_typein {
               x = 0;
            };
            y_step_typein {
               x = 0;
            };
            y_offset_typein {
               x = 0;
            };
            UIradioBoxLabel_mode1 {
               label_cmd {
                  cmd[4] = {
                     {
                        do=1,,,,,,,,,,,,,,,
                     },
                     {
                        set=1,,,,,do=1,,,,,,,,,,,,,,,
                     },
                     {
                        do=1,,,,,,,,,,,,,,,
                     },
                     {
                        do=1,,,,,,,,,,,,,,,
                     }};
               };
            };
            UIradioBoxLabel_mode2 {
               label_cmd {
                  cmd[4] = {
                     {
                        do=1,,,,,,,,,,,,,,,
                     },
                     {
                        set=1,,,,,do=1,,,,,,,,,,,,,,,
                     },
                     {
                        do=1,,,,,,,,,,,,,,,
                     },
                     {
                        do=1,,,,,,,,,,,,,,,
                     }};
               };
            };
         };
         major_line_mode = 1;
         minor_line_mode = 1;
      };
      GEOMS.Cross2D Cross2D<NEx=-44.,NEy=154.>;
      MODS.probe probe<NEx=-11.,NEy=253.> {
         in_field => <-.Mesh_plot_2D_xform.out;
         in_glyph => <-.Cross2D.out_fld;
         in_pick => <-.Uviewer2D.Scene.View.View.picked_obj;
         ProbeUI {
            probe_value {
               x = 0;
            };
         };
         ProbeParam {
            scale = 7.;
            normalize = 1;
         };
      };
      float probe_val<NEportLevels={1,2},NEx=-22.,NEy=308.,NEwidth=374.,NEheight=66.> => .probe.out_fld.node_data[0].values[0][0];
   };
   macro orthograph_UI<NEx=286.,NEy=242.,NExOffset=220.,NEyOffset=142.,NEscalingFactor=1.> {
      UImod_panel UImod_panel<NEx=-209.,NEy=-121.> {
         width = 650;
         title => "Orthograph";
         option {
            set = 1;
         };
      };
      UIframe Widget_Frame<NEx=-88.,NEy=-55.> {
         parent => <-.UImod_panel;
         y => <-.Load_File_Frame.UIframe.height;
         width => (parent.parent.width - 37);
         height => (max_array({<-.UIpanel.height,
               (<-.UIpanel#1.height + 6)}) + 15);
      };
      UIradioBox axis_radio_box<NEx=-77.,NEy=77.> {
         parent => <-.UIpanel;
         cmdList => {<-.X,<-.Y,<-.Z};
         selectedItem = 0;
         y => (<-.UIlabel.height + (((parent.height - <-.UIlabel.height) - .height) / 2));
         x => ((parent.width - 40) / 2);
         width => ((parent.width / 2) - 10);
         orientation = 0;
	 message = "Select perpendicular axis";
      };
      UIoption X<NEx=-187.,NEy=110.> {
         label => "X";
         do = 1;
         set = 1;
      };
      UIoption Y<NEx=-187.,NEy=143.> {
         do = 1;
         label => "Y";
      };
      UIoption Z<NEx=-187.,NEy=176.> {
         do = 1;
         label => "Z";
      };
      UIlabel UIlabel<NEx=-187.,NEy=44.> {
         parent => <-.UIpanel;
         label => "Axis";
         x => ((parent.width - 40) / 2);
         y = 0;
         width => parent.width;
         alignment = 0;
	 message = "Select perpendicular axis";
      };
      UIpanel UIpanel<NEx=-187.,NEy=-11.> {
         parent => <-.Widget_Frame;
         y = 0;
         width => (parent.width / 2);
         height => <-.UIpanel#1.height;
      };
      macro Load_File_Frame<NEx=-22.,NEy=-110.,NExOffset=91.,NEyOffset=45.> {
         UIfileDialog FileDialog<NEx=44.,NEy=121.,NEwidth=330.,NEheight=198.> {
            visible => <-.Load_File_Button.do;
            title => "Load pfb/pfsb File";
            isModal = 0;
            ok = 1;
            dirMaskCache = "/home/wittman/parflow/exe.SunOS/default/*";
         };
         UIbutton Load_File_Button<NEx=33.,NEy=66.> {
            x => <-.Filename_UItext.x;
            y => ((<-.Filename_UItext.y + <-.Filename_UItext.height) + 17);
            parent => <-.UIframe;
            message = "Pop up file dialog window";
            label => "Load file...";
         };
         UIframe UIframe<NEx=-77.,NEy=0.> {
            x<export=3>;
            y<export=3> = 0;
            width<export=3> => (parent.width - 10);
            parent<NEportLevels={3,0}> => <-.<-.UImod_panel;
            height<export=3> = 95;
            visible<NEportLevels={3,0}>;
         };
         UItext Filename_UItext<NEx=154.,NEy=176.> {
            x = 7;
            y = 10;
            width => (parent.width - 25);
            parent => <-.UIframe;
            height = 28;
            message = "Current File";
            text<NEportLevels={2,4},export=2> => <-.FileDialog.filename;
            rows = 1;
            resizeToText = 1;
            outputOnly = 0;
         };
      };
      UIdial X_dial<NEx=99.,NEy=0.> {
         parent => <-.UIpanel#1;
         value<NEportLevels={2,2}> = 0.;
         title => "Plane";
         y = 0;
         width => ((parent.width - 7) - 15);
         height => (.width + 40);
         min = 0.;
         max => (<-.Grid.dims[0] - 1);
         mode = 1;
         decimalPoints = 0;
         immediateMode = 0;
         showValue = 1;
         numTicks => <-.Grid.dims[0];
         visible => <-.X.set;
	 message = "Select YZ plane to view";
      };
      UIpanel UIpanel#1<NEx=165.,NEy=-66.> {
         parent => <-.Widget_Frame;
         x => <-.UIpanel.width;
         y = 0;
         width => (parent.width / 2);
         height => <-.X_dial.height;
      };
      group axis_number<NEx=0.,NEy=220.> {
         int X<NEportLevels={2,0}> => <-.X.set;
         int Y<NEportLevels={2,0}> => <-.Y.set;
         int Z<NEportLevels={2,0}> => <-.Z.set;
         int axis<NEportLevels={0,3},NEdisplayMode="open"> => switch((2 - .X),0,switch((2 - .Y),1,2));
      };
      Mesh_Struct &Grid<NEx=297.,NEy=-143.,NEportLevels={2,0}> => <-.PF_Grid_from_File.Hack;
      int plane<NEportLevels={1,2},NEx=352.,NEy=132.> => .DVswitch.out;
      UIdial Y_dial<NEx=242.,NEy=0.> {
         parent => <-.UIpanel#1;
         value<NEportLevels={2,2}> = 0.;
         title => "Plane";
         y = 0;
         width => ((parent.width - 7) - 15);
         height => (.width + 40);
         min = 0.;
         max = 14.;
         mode = 1;
         decimalPoints = 0;
         immediateMode = 0;
         showValue = 1;
         numTicks = 15;
         visible => <-.Y.set;
	 message = "Select XZ plane to view";
      };
      UIdial Z_dial<NEx=385.,NEy=0.> {
         parent => <-.UIpanel#1;
         value<NEportLevels={2,2}> = 0.;
         title => "Plane";
         y = 0;
         width => ((parent.width - 7) - 15);
         height => (.width + 40);
         min = 0.;
         max = 7.;
         mode = 1;
         decimalPoints = 0;
         immediateMode = 0;
         showValue = 1;
         numTicks = 8;
         visible => <-.Z.set;
	 message = "Select XY plane to view";
      };
      DVM.DVswitch DVswitch<NEx=286.,NEy=77.> {
         in => {<-.X_dial.value,
            <-.Y_dial.value,<-.Z_dial.value};
         index => <-.axis_number.axis;
      };
      UIframe Probe_Frame<NEx=176.,NEy=132.> {
         parent => <-.UImod_panel;
         y => (<-.Widget_Frame.y + <-.Widget_Frame.height);
         width => <-.Widget_Frame.width;
         height = 50;
      };
      UIlabel Probe_Value_Label<NEx=176.,NEy=165.> {
         parent => <-.Probe_Frame;
         label => "Probe Value";
         y => (((parent.height - height) / 2) - 3);
         width => (parent.width / 2);
	 message = "Value at current probe location (read only)";
      };
      UIfield UIfield<NEx=176.,NEy=209.> {
         parent => <-.Probe_Frame;
         value<NEportLevels={3,2}> => <-.<-.orthograph_display.probe_val;
         y => (((parent.height - height) / 2) - 3);
         x => (parent.width / 2);
         width = 75;
         outputOnly = 1;
	 message = "Value at current probe location (read only)";
      };
   };
   ParFlow.PF_Grid_from_File PF_Grid_from_File<NEx=187.,NEy=121.> {
      read_parflow {
         choose_file {
            input_file => <-.<-.<-.orthograph_UI.Load_File_Frame.Filename_UItext.text;
         };
      };
      scale_grid {
         UImod_panel {
            option {
               set = 0;
            };
         };
      };
   };
};
