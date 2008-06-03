APPS.SingleWindowApp ExcavateBrick<NExOffset=0.,NEyOffset=10.> {
   UI {
      shell {
         x = 160;
         y = 128;
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
                  <-.<-.<-.<-.ExcavateBrick_UI.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.scale_grid.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.downsize.panel.option,
                  <-.<-.<-.<-.ExcavateBrick_display.bounds.UIpanel.option,
                  <-.<-.<-.<-.ExcavateBrick_display.LegendHoriz.UImod_panel.option
               };
            };
         };
      };
   };
   ParFlow.PF_Grid_from_File PF_Grid_from_File<NEx=209.,NEy=33.> {
      read_parflow {
         choose_file {
            input_file => <-.<-.<-.ExcavateBrick_UI.Load_File_Frame.Filename_UItext.text;
         };
      };
   };
   macro ExcavateBrick_display<NEx=462.,NEy=198.,NExOffset=527.,NEyOffset=183.> {
      DVM.DVorthoslice_unif min_X_slice<NEx=-286.,NEy=-44.> {
         in => <-.Grid;
         axis = 0;
         plane => <-.min[axis];
      };
      DVM.DVorthoslice_unif min_Y_slice<NEx=-132.,NEy=-44.> {
         in => <-.Grid;
         axis = 1;
         plane => <-.min[axis];
      };
      DVM.DVorthoslice_unif min_Z_slice<NEx=33.,NEy=-44.> {
         in => <-.Grid;
         axis = 2;
         plane => <-.min[axis];
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
      MODS.bounds bounds<NEx=-495.,NEy=165.> {
         in_field => <-.Grid;
      };
      Field_Unif &Grid<NEx=-484.,NEy=-66.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      int min<NEportLevels={2,1},NEx=-440.,NEy=-110.>[3] => <-.min_slice;
      int max<NEportLevels={2,1},NEx=-385.,NEy=-165.>[3] => <-.max_slice;
      GDM.GroupObject GroupObject<NEx=-88.,NEy=187.> {
         child_objs => {
            <-.bounds.out_obj,<-.DataObject.obj,<-.DataObject#1.obj,
            <-.DataObject#2.obj,<-.extra_slices.GroupObject.obj,
            <-.excavated_slices.GroupObject.obj
         };
      };
      macro excavated_slices<NEx=-352.,NEy=110.,NExOffset=355.,NEyOffset=83.> {
         DVM.DVorthoslice_unif max_X_slice<NEx=-319.,NEy=187.> {
            in => <-.Grid;
            axis = 0;
            plane<NEdisplayMode="open"> => <-.max[.axis];
         };
         DVM.DVorthoslice_unif max_Y_slice<NEx=-99.,NEy=110.> {
            in => <-.Grid;
            axis = 1;
            plane => <-.max[.axis];
         };
         DVM.DVorthoslice_unif max_Z_slice<NEx=66.,NEy=-44.> {
            in => <-.Grid;
            axis = 2;
            plane => <-.max[.axis];
         };
         DVM.DVcrop_unif DVcrop_unif<NEx=66.,NEy=0.> {
            in => <-.max_Z_slice.out;
            min => {<-.max[0],
               <-.max[1]};
            max => {
               (<-.Grid.dims[0] - 1),(<-.Grid.dims[1] - 1)};
         };
         DVM.DVcrop_unif DVcrop_unif#1<NEx=-99.,NEy=154.> {
            in => <-.max_Y_slice.out;
            min => {<-.max[0],
               <-.max[2]};
            max<NEdisplayMode="open"> => {
               (<-.Grid.dims[0] - 1),(<-.Grid.dims[2] - 1)};
         };
         DVM.DVcrop_unif DVcrop_unif#2<NEx=-319.,NEy=231.> {
            in => <-.max_X_slice.out;
            min => {<-.max[1],
               <-.max[2]};
            max => {
               (<-.Grid.dims[1] - 1),(<-.Grid.dims[2] - 1)};
         };
         GDM.DataObject DataObject<NEx=66.,NEy=44.> {
            in => <-.DVcrop_unif.out;
         };
         GDM.DataObject DataObject#1<NEx=-99.,NEy=198.> {
            in => <-.DVcrop_unif#1.out;
         };
         GDM.DataObject DataObject#2<NEx=-319.,NEy=275.> {
            in => <-.DVcrop_unif#2.out;
         };
         Field_Unif &Grid<NEx=-341.,NEy=11.,NEportLevels={2,1}> => <-.Grid;
         int min<NEportLevels={2,1},NEx=-176.,NEy=-44.>[3] => <-.min;
         int max<NEportLevels={2,1},NEx=-176.,NEy=-77.>[3] => <-.max;
         GDM.GroupObject GroupObject<NEx=198.,NEy=275.> {
            child_objs => {<-.DataObject.obj,
               <-.DataObject#1.obj,<-.DataObject#2.obj};
            obj<NEportLevels={1,3}>;
         };
      };
      macro extra_slices<NEx=-242.,NEy=66.,NExOffset=353.,NEyOffset=101.> {
         DVM.DVorthoslice_unif max_X_slice<NEx=-22.,NEy=-77.> {
            in => <-.Grid;
            axis = 0;
            plane => (in.dims[.axis] - 1);
         };
         DVM.DVorthoslice_unif max_Y_slice<NEx=-143.,NEy=33.> {
            in => <-.Grid;
            axis = 1;
            plane => (in.dims[.axis] - 1);
         };
         DVM.DVorthoslice_unif max_Z_slice<NEx=-330.,NEy=143.> {
            in => <-.Grid;
            axis = 2;
            plane => (in.dims[.axis] - 1);
         };
         DVM.DVcrop_unif DVcrop_unif<NEx=-330.,NEy=187.> {
            in => <-.max_Z_slice.out;
            min => {<-.min[0],
               <-.min[1]};
            max => {
               (<-.Grid.dims[0] - 1),<-.max[1]};
         };
         GDM.DataObject DataObject<NEx=-330.,NEy=231.> {
            in => <-.DVcrop_unif.out;
         };
         Field_Unif &Grid<NEx=-341.,NEy=-11.,NEportLevels={2,1}> => <-.Grid;
         int min<NEportLevels={2,1},NEx=-220.,NEy=-44.>[3] => <-.min;
         int max<NEportLevels={2,1},NEx=-220.,NEy=-77.>[3] => <-.max;
         DVM.DVcrop_unif DVcrop_unif#1<NEx=-176.,NEy=187.> {
            in => <-.max_Z_slice.out;
            min => {<-.min[0],
               <-.max[1]};
            max => {<-.max[0],
               (<-.Grid.dims[1] - 1)};
         };
         GDM.DataObject DataObject#1<NEx=-176.,NEy=231.> {
            in => <-.DVcrop_unif#1.out;
         };
         DVM.DVcrop_unif DVcrop_unif#2<NEx=77.,NEy=-22.> {
            in => <-.max_X_slice.out;
            min => {<-.min[1],
               <-.min[2]};
            max => {
               (<-.Grid.dims[1] - 1),<-.max[2]};
         };
         GDM.DataObject DataObject#2<NEx=77.,NEy=22.> {
            in => <-.DVcrop_unif#2.out;
         };
         DVM.DVcrop_unif DVcrop_unif#3<NEx=231.,NEy=-77.> {
            in => <-.max_X_slice.out;
            min => {<-.min[1],
               <-.max[2]};
            max => {<-.max[1],
               (<-.Grid.dims[2] - 1)};
         };
         GDM.DataObject DataObject#3<NEx=231.,NEy=-33.> {
            in => <-.DVcrop_unif#3.out;
         };
         DVM.DVcrop_unif DVcrop_unif#4<NEx=-132.,NEy=88.> {
            in => <-.max_Y_slice.out;
            min => {<-.min[0],
               <-.min[2]};
            max => {
               (<-.Grid.dims[0] - 1),<-.max[2]};
         };
         GDM.DataObject DataObject#4<NEx=-132.,NEy=132.> {
            in => <-.DVcrop_unif#4.out;
         };
         DVM.DVcrop_unif DVcrop_unif#5<NEx=11.,NEy=88.> {
            in => <-.max_Y_slice.out;
            min => {<-.min[0],
               <-.max[2]};
            max => {<-.max[0],
               (<-.Grid.dims[2] - 1)};
         };
         GDM.DataObject DataObject#5<NEx=11.,NEy=132.> {
            in => <-.DVcrop_unif#5.out;
         };
         GDM.GroupObject GroupObject<NEx=220.,NEy=253.> {
            child_objs => {
               <-.DataObject.obj,<-.DataObject#1.obj,<-.DataObject#4.obj,
               <-.DataObject#5.obj,<-.DataObject#2.obj,<-.DataObject#3.obj
            };
            obj<NEportLevels={1,3}>;
         };
      };
      GEOMS.LegendHoriz LegendHoriz<NEx=88.,NEy=132.> {
         obj_in => <-.DataObject#3.obj;
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
      GDM.DataObject DataObject#3<NEx=11.,NEy=88.> {
         in => <-.Grid;
      };
      GDM.Uviewer Uviewer<NEx=88.,NEy=198.> {
         Scene {
            Top {
               child_objs => {
                  <-.<-.<-.GroupObject.obj};
               Xform {
                  mat = {
                     -0.0325533,-0.00856045,0.0330307,0.,0.0341219,-0.00810985,0.031527,
0.,-4.26829e-05,0.0456615,0.0117919,0.,0.,0.,0.,1.
                  };
                  xlate = {-1.84196,3.56968,
-3.29746};
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
   };
   int max_slice<NEportLevels=1,NEx=242.,NEy=176.,NEwidth=308.,NEheight=143.>[3] => {
      .ExcavateBrick_UI.X_Slice,.ExcavateBrick_UI.Y_Slice,
      .ExcavateBrick_UI.Z_Slice};
   int min_slice<NEportLevels={1,1},NEx=242.,NEy=143.>[3] = {0,0,0};
   ParFlow.Slice3_UI ExcavateBrick_UI<NEx=44.,NEy=88.,NExOffset=454.,NEyOffset=220.> {
      UImod_panel {
         title = "ExcavateBrick";
      };
      X_Dial {
         min = 1.;
         max => (<-.Grid.dims[0] - 2);
      };
      Grid => <-.PF_Grid_from_File.Hack;
      Y_Dial {
         min = 1.;
         max => (<-.Grid.dims[1] - 2);
      };
      X_Slider {
         min = 1.;
         max => (<-.Grid.dims[0] - 2);
      };
      Y_Slider {
         min = 1.;
         max => (<-.Grid.dims[1] - 2);
      };
      Z_Slider {
         min = 1.;
         max => (<-.Grid.dims[2] - 2);
      };
      X_Slice = 1;
      Y_Slice = 1;
      Z_Slice = 1;
      Z_dial {
         min = 1.;
         max => (<-.Grid.dims[2] - 2);
      };
      Load_File_Frame {
         FileDialog<NEx=341.,NEy=110.>;
      };
   };
};
