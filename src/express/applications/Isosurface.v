APPS.SingleWindowApp Isosurface {
   UI {
      shell {
         x = 58;
         y = 156;
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
            panel {
               visible = 1;
            };
            optionList {
               selectedItem = 0;
               cmdList => {
                  <-.<-.<-.<-.Isosurface_UI.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.scale_grid.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.downsize.panel.option,
                  <-.<-.<-.<-.Isosurface_display.bounds.UIpanel.option,
                  <-.<-.<-.<-.Isosurface_display.Axis3D.UIpanel.option,
                  <-.<-.<-.<-.Isosurface_display.isosurface.IsoUI.UIpanel.option,
                  <-.<-.<-.<-.Isosurface_display.LegendHoriz.UImod_panel.option
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
   macro Isosurface_display<NEx=440.,NEy=253.,NExOffset=294.,NEyOffset=164.> {
      MODS.bounds bounds<NEx=-286.,NEy=-33.> {
         in_field => <-.Field_Unif;
      };
      Field_Unif &Field_Unif<NEx=-231.,NEy=-132.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      MODS.isosurface isosurface<NEx=-143.,NEy=-33.> {
         in_field => <-.Field_Unif;
         IsoParam {
            iso_level<NEportLevels={4,2}> => <-.<-.<-.Isosurface_UI.level;
         };
         Iso {
            DVnmap {
               out {
                  nnode_data = 1;
               };
            };
         };
      };
      GEOMS.Axis3D Axis3D<NEx=0.,NEy=-33.> {
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
      GEOMS.LegendHoriz LegendHoriz<NEx=297.,NEy=-33.> {
         obj_in<NEdisplayMode="showParams"> => <-.DataObject.obj;
         LabelsMacro {
            labelsOffset = 0.1;
         };
         TicksMacro {
            Ticks {
               visible = 0;
            };
         };
         x_min = -0.9;
         x_max = 0.9;
         y_min = -0.95;
         y_max = -0.9;
         numIntervals = 2;
      };
      GDM.Uviewer Uviewer<NEx=-176.,NEy=121.> {
         Scene {
            Top {
               child_objs => {
                  <-.<-.<-.isosurface.out_obj,<-.<-.<-.bounds.out_obj,
                  <-.<-.<-.Axis3D.out_obj};
               Xform {
                  mat = {
                     -0.0271257,-0.0106662,0.0376588,0.,0.0391399,-0.00757494,0.0260469,
0.,0.000156249,0.0457887,0.0130815,0.,0.,0.,0.,1.
                  };
                  xlate = {-1.52844,-1.84713,
-5.49551};
               };
            };
            Top2D {
               child_objs => {
                  <-.<-.<-.LegendHoriz.GroupObject.obj};
            };
            Camera2D {
               Camera {
                  pickable = 0;
                  extents = 1;
               };
            };
         };
         Scene_Editor {
            Camera_Editor {
               GDcamera_edit {
                  pickable = 1;
                  extents = 0;
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
      GDM.DataObject DataObject<NEx=143.,NEy=-33.> {
         in => <-.Field_Unif;
      };
   };
   ParFlow.PF_Grid_from_File PF_Grid_from_File<NEx=319.,NEy=77.> {
      read_parflow {
         choose_file {
            input_file => <-.<-.<-.Isosurface_UI.Load_File_Frame.Filename_UItext.text;
         };
         ReadPFB {
            Uniform_Grid {
               node_data<NEdisplayMode="closed">;
            };
         };
      };
      sparse_to_unif {
         out {
            node_data<NEdisplayMode="open">;
         };
      };
   };
   macro Isosurface_UI<NEx=77.,NEy=154.,NExOffset=0.,NEyOffset=-11.> {
      Field_Unif &Grid<NEx=66.,NEy=33.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      UImod_panel UImod_panel<NEx=44.,NEy=99.> {
         title => "Isosurface^2";
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
         title => "Iso-level";
         mode = 2;
         decimalPoints = 4;
         message = "Isosurface level";
         immediateMode = 0;
         showValue = 1;
         numTicks = 15;
         min => <-.min.output;
         max => (<-.max.output + (<-.min.output == <-.max.output));
      };
      UIpanel UIpanel<NEx=110.,NEy=198.> {
         parent => <-.IsoFrame;
         y = 0;
         width => parent.width;
         height => (<-.Min_field.y + <-.Min_field.height);
      };
      ParFlow.Load_File_Frame Load_File_Frame<NEx=330.,NEy=66.> {
         UIframe {
            parent => <-.<-.UImod_panel;
         };
         Filename_UItext {
            text<NEportLevels={2,4}>;
            height = 33;
         };
      };
      float level<NEportLevels={0,2},NEx=77.,NEy=352.> = 0.001;
      GMOD.copy_on_change min<NEx=473.,NEy=165.> {
         input => <-.Grid.node_data[0].min;
         output = -1000.1;
      };
      GMOD.copy_on_change max<NEx=473.,NEy=198.> {
         input => <-.Grid.node_data[0].max;
         output = 1000.1;
      };
   };
};
