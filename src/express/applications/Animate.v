APPS.SingleWindowApp Animate {
   UI {
      shell {
         x = 58;
         y = 151;
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
               selectedItem = 0;
               cmdList => {
                  <-.<-.<-.<-.Animate_UI.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.scale_grid.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.downsize.panel.option,
                  <-.<-.<-.<-.Animate_display.bounds.UIpanel.option,
                  <-.<-.<-.<-.Animate_display.Axis3D.UIpanel.option,
                  <-.<-.<-.<-.Animate_display.LegendHoriz.UImod_panel.option,
                  <-.<-.<-.<-.Animate_display.isosurface.IsoUI.UIpanel.option
               };
            };
         };
      };
      Windows {
         IUI {
            optionList {
               selectedItem = 0;
            };
         };
      };
   };
   macro Animate_display<NEx=440.,NEy=253.,NExOffset=294.,NEyOffset=164.> {
      MODS.bounds bounds<NEx=-110.,NEy=-33.> {
         in_field => <-.Field_Unif;
      };
      Field_Unif &Field_Unif<NEx=-231.,NEy=-132.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      MODS.isosurface isosurface<NEx=33.,NEy=-33.> {
         in_field => <-.Field_Unif;
         IsoParam {
            iso_level<NEportLevels={4,2}> => <-.<-.<-.Animate_UI.level;
         };
         Iso {
            DVnmap {
               out {
                  nnode_data = 1;
               };
            };
         };
         DVcell_data_labels {
            labels[];
         };
         DVnode_data_labels {
            labels[];
         };
      };
      GEOMS.Axis3D Axis3D<NEx=176.,NEy=-33.> {
         in_field => <-.Field_Unif;
         x_axis_param {
            minor_ticks = 0;
            off_anno => 1.;
         };
         y_axis_param {
            minor_ticks = 0;
            off_anno => 1.;
         };
         z_axis_param {
            minor_ticks = 0;
            off_anno => 1.;
         };
         Axis3DUI {
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
            z_start_typein {
               x = 0;
            };
            z_end_typein {
               x = 0;
            };
            z_origin_typein {
               x = 0;
            };
            z_step_typein {
               x = 0;
            };
            z_offset_typein {
               x = 0;
            };
            UIradioBoxLabel_mode1 {
               label_cmd {
                  cmd[4] = {,
                     {
                        do=1,,,,,,,,,,,,,,,
                     },
                     {
                        set=1,,,,,do=1,,,,,,,,,,,,,,,
                     },
                     {
                        do=1,,,,,,,,,,,,,,,
                     }};
               };
            };
         };
         major_line_mode = 2;
         obj_y {
            PickInfo {
               objects => {Axis3D.obj.Top,Obj};
               nobjs = 2;
            };
         };
      };
      GEOMS.LegendHoriz LegendHoriz<NEx=319.,NEy=11.> {
         obj_in => <-.DataObject.obj;
         LabelsMacro {
            labelsOffset = 0.1;
         };
         TicksMacro {
            Ticks {
               visible = 0;
            };
         };
         y_min = -0.95;
         y_max = -0.9;
         numIntervals = 2;
      };
      GDM.Uviewer Uviewer<NEx=0.,NEy=121.> {
         Scene {
            Top {
               child_objs => {
                  <-.<-.<-.isosurface.out_obj,<-.<-.<-.bounds.out_obj,
                  <-.<-.<-.Axis3D.out_obj};
               Xform {
                  mat = {
                     -0.0216786,-0.00852432,0.0300965,0.,0.0312802,-0.00605381,0.0208164,
0.,0.000124872,0.0365938,0.0104546,0.,0.,0.,0.,1.
                  };
                  xlate = {-1.22244,0.899411,
-3.72442};
               };
            };
            Top2D {
               child_objs => {
                  <-.<-.<-.LegendHoriz.GroupObject.obj};
            };
            Camera2D {
               Camera {
                  pickable = 0;
                  extents = "Window";
               };
            };
            View {
               View {
                  trigger = 1;
               };
            };
         };
         Scene_Editor {
            Camera_Editor {
               GDcamera_edit {
                  pickable = 1;
                  extents = "Compute";
               };
               IUI {
                  General {
                     IUI {
                        Extents {
                           OPcmdList = {
                              {
                                 set=1,,,,,,,,,,,,,,,,,,,,
                              },
                              };
                        };
                     };
                  };
               };
            };
         };
      };
      GDM.DataObject DataObject<NEx=319.,NEy=-33.> {
         in => <-.Field_Unif;
      };
   };
   PF_Grid_from_File PF_Grid_from_File<NEx=319.,NEy=77.> {
      read_parflow {
         choose_file {
            input_file => <-.<-.<-.loop_pfsb_files.current_file;
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
   macro Animate_UI<NEx=33.,NEy=154.,NExOffset=-29.,NEyOffset=-18.> {
      Field_Unif &Grid<NEx=66.,NEy=33.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      UImod_panel UImod_panel<NEx=44.,NEy=99.> {
         title => "Animate";
         width = 700;
         option {
            set = 1;
         };
      };
      UIframe IsoFrame<NEx=44.,NEy=143.> {
         parent => <-.UImod_panel;
         y = 95;
         width => (parent.parent.width - 36);
         height => ((<-.UIdial.y + <-.UIdial.height) + 20);
      };
      UIlabel Min_label<NEx=44.,NEy=264.> {
         parent => <-.UIpanel;
         label => "Minimum";
         x => (((parent.width / 2) - .width) / 2);
         y = 0;
         message = "Minimum grid value";
      };
      UIfield Min_field<NEx=44.,NEy=297.> {
         parent => <-.UIpanel;
         y = 24;
         x => (((parent.width / 2) - .width) / 2);
         decimalPoints = 8;
         outputOnly = 1;
         value => <-.Grid.node_data[0].min;
         message = "Minimum grid value";
      };
      UIlabel Max_label<NEx=187.,NEy=264.> {
         parent => <-.UIpanel;
         label => "Maximum";
         y = 0;
         x => (((parent.width / 2) + (((parent.width / 2) - .width) / 2)) - 5);
         message = "Maximum grid value";
      };
      UIfield Max_field<NEx=187.,NEy=297.> {
         parent => <-.UIpanel;
         y = 24;
         value<NEportLevels={2,2}> => <-.Grid.node_data[0].max;
         x => (((parent.width / 2) + (((parent.width / 2) - .width) / 2)) - 5);
         decimalPoints = 8;
         outputOnly = 1;
         message = "Maximum grid value";
      };
      UIdial UIdial<NEx=242.,NEy=374.> {
         parent => <-.IsoFrame;
         value => <-.level;
         x => (((parent.width - .width) / 2) - 4);
         width => (parent.width / 2.5);
         y => (<-.UIpanel.height + 15);
         height => (.width + 40);
         min => <-.min.output;
         max => (<-.max.output + (<-.min.output == <-.max.output));
         title => "Iso-level";
         mode = "real";
         decimalPoints = 4;
         message = "Isosurface level";
         immediateMode = 0;
         showValue = 1;
         numTicks = 15;
      };
      UIpanel UIpanel<NEx=110.,NEy=198.> {
         parent => <-.IsoFrame;
         y = 0;
         width => parent.width;
         height => (<-.Min_field.y + <-.Min_field.height);
      };
      strsplit_rev split_ext<NEx=517.,NEy=55.> {
         string1<NEportLevels={3,0}>;
         string2 = ".";
      };
      strsplit_rev split_seq_num<NEx=517.,NEy=99.> {
         string1 => <-.split_ext.outstring1;
         string2 = ".";
      };
      Load_File_Frame Load_File_Frame<NEx=242.,NEy=77.> {
         UIframe {
            parent => <-.<-.UImod_panel;
         };
         Filename_UItext {
            text<NEportLevels={2,4}>;
            height = 33;
         };
         FileDialog {
            filename = "/tmp_mnt/home/ssmith/parflow/exe.IRIX64/test/default_single.out.concen.0.00.00000.pfsb";
            title = "Load pfsb File";
            searchPattern = "*.pfsb";
            x = 490;
            y = 267;
            width = 347;
            height = 390;
            dirMaskCache = "/tmp_mnt/home/ssmith/parflow/exe.IRIX64/test/*";
         };
         Load_File_Button {
            label = "Select file in sequence...";
            width = 160;
         };
      };
      float level<NEportLevels={0,2},NEx=77.,NEy=352.> = 0.1013;
      UIframe AnimateFrame<NEx=451.,NEy=154.> {
         parent => <-.UImod_panel;
         y = 324;
         width => (parent.parent.width - 36);
         height => ((<-.start_field.y + <-.start_field.height) + 10);
      };
      UIlabel number_label<NEx=451.,NEy=198.> {
         parent => <-.AnimateFrame;
         label => ("Current file is " + <-.split_seq_num.outstring2);
         y = 0;
         width => (parent.width - 10);
         message = "Sequence number of current file";
      };
      UItoggle Reset<NEx=396.,NEy=253.> {
         parent => <-.AnimateFrame;
         label => "Reset";
         set<NEportLevels={0,3}>;
         y => <-.number_label.height;
         width = 80;
         message = "Click to reset to start file";
      };
      UItoggle Run<NEx=539.,NEy=253.> {
         parent => <-.AnimateFrame;
         label => "Run";
         do = 1;
         set<NEportLevels={0,3}>;
         x = 80;
         y => <-.number_label.height;
         width = 80;
         message = "Click to run animation";
      };
      UItoggle Cycle<NEx=484.,NEy=297.> {
         parent => <-.AnimateFrame;
         label => "Cycle";
         set<NEportLevels={0,3}>;
         y => <-.number_label.height;
         x = 160;
         width = 80;
         message = "Set toggle to continuously cyle animation run";
      };
      UIlabel start_label<NEx=407.,NEy=352.> {
         parent => <-.AnimateFrame;
         label => "start";
         y => (<-.Reset.height + <-.Reset.y);
         width => (parent.width / 3);
         message = "Initial animation file";
      };
      UIfield start_field<NEx=407.,NEy=385.> {
         parent => <-.AnimateFrame;
         value = 0.;
         y => (<-.start_label.y + <-.start_label.height);
         width => (parent.width / 3);
         min = 0.;
         max => max_array({(<-.files - 1),0});
         mode = "integer";
         updateMode = 7;
         message = "Initial animation file";
      };
      UIlabel end_label<NEx=550.,NEy=352.> {
         parent => <-.AnimateFrame;
         label => "end";
         y => (<-.Reset.height + <-.Reset.y);
         width => (parent.width / 3);
         x => (parent.width / 3);
         message = "Last animation file";
      };
      UIfield end_field<NEx=550.,NEy=385.> {
         parent => <-.AnimateFrame;
         y => (<-.end_label.y + <-.end_label.height);
         width => (parent.width / 3);
         value<NEportLevels={2,2}> => <-.copy_on_change.output;
         max => max_array({(<-.files - 1),0});
         x => (parent.width / 3);
         min = 0.;
         mode = "integer";
         updateMode = 7;
         message = "Last animation file";
      };
      UIlabel stride_label<NEx=484.,NEy=429.> {
         parent => <-.AnimateFrame;
         label => "stride";
         y => (<-.Reset.height + <-.Reset.y);
         width => (parent.width / 3);
         x => (2 * (parent.width / 3));
         message = "File number increment at each step";
      };
      UIfield stride_field<NEx=484.,NEy=462.> {
         parent => <-.AnimateFrame;
         y => (<-.stride_label.y + <-.stride_label.height);
         width => (parent.width / 3);
         value<NEportLevels={2,2}> = 1.;
         x => (2 * (parent.width / 3));
         min = 1.;
         mode = "integer";
         max => max_array({(<-.files - 1),1});
         updateMode = 7;
         message = "File number increment at each step";
      };
      int files<NEportLevels={2,1},NEx=308.,NEy=33.> => <-.loop_pfsb_files.files;
      int start<NEportLevels={1,2},NEx=352.,NEy=506.> => .start_field.value;
      int end<NEportLevels={1,2},NEx=352.,NEy=539.> => .end_field.value;
      int stride<NEportLevels={1,2},NEx=352.,NEy=572.> => .stride_field.value;
      GMOD.copy_on_change copy_on_change<NEx=638.,NEy=308.> {
         trigger = 0;
         input => <-.end_field.max;
      };
      GMOD.copy_on_change min<NEx=275.,NEy=154.> {
         output = -1000.1;
      };
      GMOD.copy_on_change max<NEx=275.,NEy=187.> {
         output = 1000.1;
      };
   };
   macro loop_pfsb_files<NEx=88.,NEy=308.,NExOffset=135.,NEyOffset=35.,NEdisplayMode="maximized"> {
      GMOD.shell_command shell_command<NEx=99.,NEy=187.,process="express"> {
         command => (("ls " + <-.basename) + ".*.pfsb");
      };
      strsplit_all split_lines<NEy=242.> {
         string1 => <-.shell_command.stdout_string;
         string2 = "\n";
      };
      pfsb_series_basename pfsb_series_basename<NEx=-110.,NEy=154.> {
         strsplit_rev {
            string1<NEportLevels={4,0}> => <-.<-.<-.Animate_UI.Load_File_Frame.Filename_UItext.text;
         };
      };
      GMOD.loop loop<NEx=-66.,NEy=99.,process="express"> {
         reset<NEportLevels={3,2}> => <-.<-.Animate_UI.Reset.set;
         run<NEportLevels={3,0}> => <-.<-.Animate_UI.Run.set;
         cycle<NEportLevels={3,0}> => <-.<-.Animate_UI.Cycle.set;
         start_val => <-.start;
         end_val => switch((2 - (<-.end < 0)),0,<-.end);
         incr => <-.incr;
         count = 0.;
      };
      string basename<NEportLevels=1,NEx=-88.,NEy=220.> => .pfsb_series_basename.strsplit_rev#1.outstring1;
      int start<NEportLevels={2,1},NEx=121.,NEy=-33.> => <-.Animate_UI.start;
      int end<NEportLevels={2,1},NEx=121.,NEy=0.> => <-.Animate_UI.end;
      int incr<NEportLevels={2,1},NEx=121.,NEy=33.> => <-.Animate_UI.stride;
      string current_file<NEportLevels={1,2},NEx=231.,NEy=308.,NEwidth=616.,NEheight=66.> => split_lines.array[loop.count];
      int files<NEportLevels={1,2},NEx=-88.,NEy=308.> => (split_lines.strings - 1);
   };
};
