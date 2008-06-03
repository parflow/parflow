APPS.SingleWindowApp Brick<NExOffset=0.,NEyOffset=10.> {
   UI {
      shell {
         x = 4;
         y = 141;
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
            panel {
               visible = 1;
            };
            optionList {
               selectedItem = 0;
               cmdList => {
                  <-.<-.<-.<-.Brick_UI.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.scale_grid.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.downsize.panel.option,
                  <-.<-.<-.<-.Brick_display.bounds.UIpanel.option,
                  <-.<-.<-.<-.Brick_display.LegendHoriz.UImod_panel.option
               };
            };
         };
      };
   };
   ParFlow.PF_Grid_from_File PF_Grid_from_File<NEx=209.,NEy=33.> {
      read_parflow {
         choose_file {
            input_file => <-.<-.<-.Brick_UI.Load_File_Frame.Filename_UItext.text;
         };
      };
   };
   macro Brick_display<NEx=462.,NEy=198.,NExOffset=527.,NEyOffset=183.> {
      DVM.DVorthoslice_unif min_X_slice<NEx=-286.,NEy=-44.> {
         in => <-.Grid;
         axis = 0;
         plane = 0;
      };
      DVM.DVorthoslice_unif min_Y_slice<NEx=-132.,NEy=-44.> {
         in => <-.Grid;
         axis = 1;
         plane = 0;
      };
      DVM.DVorthoslice_unif min_Z_slice<NEx=33.,NEy=-44.> {
         in => <-.Grid;
         axis = 2;
         plane = 0;
      };
      GDM.DataObject DataObject<NEx=-286.,NEy=0.> {
         in => <-.min_X_slice.out;
      };
      GDM.DataObject DataObject#1<NEx=-132.,NEy=0.> {
         in => <-.min_Y_slice.out;
      };
      GDM.DataObject DataObject#2<NEx=33.,NEy=0.> {
         in => <-.min_Z_slice.out;
      };
      MODS.bounds bounds<NEx=-473.,NEy=154.> {
         in_field => <-.Grid;
      };
      DVM.DVcrop_unif crop<NEx=-484.,NEy=0.,NEwidth=209.,NEheight=242.> {
         in => <-.Grid;
         min => max_array({<-.min,
               {0,0,0}});
         max<NEdisplayMode="open"> => min_array({<-.max,
               (in.dims - {1,1,1})});
      };
      DVM.DVorthoslice_unif max_X_slice<NEx=-297.,NEy=77.> {
         in => <-.crop.out;
         axis = 0;
         plane => (in.dims[.axis] - 1);
      };
      DVM.DVorthoslice_unif max_Y_slice<NEx=-132.,NEy=77.> {
         in => <-.crop.out;
         axis = 1;
         plane => (in.dims[.axis] - 1);
      };
      DVM.DVorthoslice_unif max_Z_slice<NEx=33.,NEy=77.> {
         in => <-.crop.out;
         axis = 2;
         plane => (in.dims[.axis] - 1);
      };
      GDM.DataObject DataObject#3<NEx=-297.,NEy=121.> {
         in => <-.max_X_slice.out;
      };
      GDM.DataObject DataObject#4<NEx=-132.,NEy=121.> {
         in => <-.max_Y_slice.out;
      };
      GDM.DataObject DataObject#5<NEx=33.,NEy=121.> {
         in => <-.max_Z_slice.out;
      };
      Field_Unif &Grid<NEx=-495.,NEy=99.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      int min_X<NEportLevels=1,NEx=-110.,NEy=-176.> = 1;
      int min_Y<NEportLevels={1,1},NEx=-110.,NEy=-143.> = 2;
      int min_Z<NEportLevels={1,1},NEx=-110.,NEy=-110.> = 3;
      int max_X<NEportLevels={1,1},NEx=55.,NEy=-176.>;
      int max_Y<NEportLevels={1,1},NEx=55.,NEy=-143.>;
      int max_Z<NEportLevels={1,1},NEx=55.,NEy=-110.>;
      int min<NEportLevels={2,1},NEx=-484.,NEy=-121.>[3] => <-.min_slice;
      int max<NEportLevels={2,1},NEx=-319.,NEy=-154.>[3] => <-.max_slice;
      GDM.GroupObject GroupObject<NEx=-55.,NEy=165.> {
         child_objs => {
            <-.bounds.out_obj,<-.DataObject#3.obj,<-.DataObject#4.obj,
            <-.DataObject#5.obj,<-.DataObject.obj,<-.DataObject#1.obj,
            <-.DataObject#2.obj
         };
      };
      GDM.Uviewer Uviewer<NEx=77.,NEy=220.> {
         Scene {
            Top {
               child_objs => {
                  <-.<-.<-.GroupObject.obj};
               Xform {
                  mat = {
                     -0.0281197,-0.0118727,0.0357853,0.,0.0376828,-0.00737519,0.0271638,
0.,-0.00124554,0.0449101,0.0139212,0.,0.,0.,0.,1.
                  };
                  xlate = {-4.33775,-5.59603,
-4.75757};
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
      GEOMS.LegendHoriz LegendHoriz<NEx=-242.,NEy=209.> {
         obj_in => <-.DataObject#6.obj;
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
      GDM.DataObject DataObject#6<NEx=-495.,NEy=209.> {
         in => <-.Grid;
      };
   };
   int max_slice<NEportLevels=1,NEx=242.,NEy=176.,NEwidth=308.,NEheight=143.>[3] => {.Brick_UI.X_Slice,
      .Brick_UI.Y_Slice,.Brick_UI.Z_Slice};
   int min_slice<NEportLevels={1,1},NEx=242.,NEy=143.>[3] = {0,0,0};
   ParFlow.Slice3_UI Brick_UI<NEx=44.,NEy=88.,NExOffset=454.,NEyOffset=226.> {
      UImod_panel {
         title = "Brick";
      };
      X_Dial {
         min = 1.;
      };
      Y_Dial {
         min = 1.;
      };
      Grid => <-.PF_Grid_from_File.Hack;
      X_Slider {
         min = 1.;
      };
      Y_Slider {
         min = 1.;
      };
      Z_Slider {
         min = 1.;
      };
      X_Slice = 1;
      Y_Slice = 1;
      Z_Slice = 1;
      Z_dial {
         min = 1.;
      };
   };
};
