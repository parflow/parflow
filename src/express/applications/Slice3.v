APPS.SingleWindowApp Slice3<NExOffset=-221.,NEyOffset=-175.,NEscalingFactor=1.> {
   UI {
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
               cmdList => {
                  <-.<-.<-.<-.Slice3_UI.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.scale_grid.UImod_panel.option,
                  <-.<-.<-.<-.PF_Grid_from_File.downsize.panel.option,
                  <-.<-.<-.<-.Slice3_display.bounds.UIpanel.option,
                  <-.<-.<-.<-.Slice3_display.LegendHoriz.UImod_panel.option
               };
               selectedItem = 0;
            };
         };
      };
      shell {
         x = 0;
         y = 185;
      };
      Windows {
         IUI {
            optionList {
               selectedItem = 0;
            };
         };
      };
   };
   macro Slice3_display<NEx=616.,NEy=418.,NExOffset=252.,NEyOffset=89.> {
      GDM.DataObject DataObject#2<NEx=319.,NEy=99.> {
         in => <-.Z_Plane_Slicer.out;
      };
      GDM.DataObject DataObject#1<NEx=11.,NEy=77.> {
         in => <-.X_Plane_Slicer.out;
      };
      DVM.DVorthoslice_unif X_Plane_Slicer<NEx=0.,NEy=22.> {
         in => <-.Field_Unif;
         axis = 0;
         plane<NEportLevels={3,0}> => <-.<-.Slice3_UI.X_Slice;
      };
      DVM.DVorthoslice_unif Y_Plane_Slicer<NEx=154.,NEy=22.> {
         in => <-.Field_Unif;
         plane<NEportLevels={3,0}> => <-.<-.Slice3_UI.Y_Slice;
         axis = 1;
      };
      DVM.DVorthoslice_unif Z_Plane_Slicer<NEx=319.,NEy=22.> {
         axis = 2;
         in => <-.Field_Unif;
         plane<NEportLevels={3,0}> => <-.<-.Slice3_UI.Z_Slice;
      };
      GDM.DataObject DataObject#3<NEx=165.,NEy=77.> {
         in => <-.Y_Plane_Slicer.out;
      };
      Field_Unif &Field_Unif<NEx=-143.,NEy=-55.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      MODS.bounds bounds<NEx=-220.,NEy=44.> {
         in_field => <-.Field_Unif;
      };
      GDM.Uviewer Uviewer<NEx=-198.,NEy=264.> {
         Scene {
            Top {
               child_objs => {
                  <-.<-.<-.bounds.out_obj,<-.<-.<-.DataObject#1.obj,
                  <-.<-.<-.DataObject#3.obj,<-.<-.<-.DataObject#2.obj};
               Xform {
                  mat = {
                     -0.0310311,-0.00869048,0.0337542,0.,0.0348375,-0.00630001,0.0304049,
0.,-0.00110536,0.0454157,0.0106768,0.,0.,0.,0.,1.
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
      GEOMS.LegendHoriz LegendHoriz<NEx=132.,NEy=286.> {
         obj_in => <-.DataObject.obj;
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
      GDM.DataObject DataObject<NEx=132.,NEy=242.> {
         in => <-.Field_Unif;
      };
   };
   macro Slice3_UI<NEx=561.,NEy=330.,NExOffset=462.,NEyOffset=232.> {
      UImod_panel UImod_panel<NEx=-429.,NEy=-187.> {
         title => "Slice3";
         option {
            set = 1;
         };
         width = 700;
      };
      macro Load_File_Frame<NEx=-198.,NEy=-209.,NExOffset=91.,NEyOffset=45.> {
         UIfileDialog FileDialog<NEx=319.,NEy=121.,NEwidth=330.,NEheight=198.> {
            visible => <-.Load_File_Button.do;
            title => "Load pfb/pfsb File";
            isModal = 0;
            ok = 1;
            dirMaskCache = "/home/wittman/parflow/exe.SunOS/default/*";
         };
         UIbutton Load_File_Button<NEx=330.,NEy=55.> {
            x => <-.Filename_UItext.x;
            y => ((<-.Filename_UItext.y + <-.Filename_UItext.height) + 17);
            parent => <-.UIframe;
            message = "Pop up file dialog window";
            label => "Load file...";
         };
         UIframe UIframe<NEx=-33.,NEy=0.> {
            y = 0;
            width => (parent.width - 10);
            parent<NEportLevels={3,0}> => <-.<-.UImod_panel;
            height = 95;
            visible<NEportLevels={3,0}>;
         };
         UItext Filename_UItext<NEx=132.,NEy=55.> {
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
         int width<NEportLevels={1,1},NEx=77.,NEy=198.,export=2> => .UIframe.width;
         int height<NEportLevels={1,1},NEx=77.,NEy=231.,export=2> => .UIframe.height;
         int x<NEportLevels={1,1},NEx=77.,NEy=264.,export=2> => .UIframe.x;
         int y<NEportLevels={1,1},NEx=77.,NEy=297.,export=2> => .UIframe.y;
      };
      UIdial X_Dial<NEx=-132.,NEy=165.> {
         parent => <-.Dial_Panel;
         y = 0;
         width => (parent.width / 3);
         height => (.width + 40);
         title => "X Plane";
         mode = 1;
         decimalPoints = 0;
         min = 0.;
         max => (<-.Grid.dims[0] - 1);
         message = "Select x slice";
         immediateMode = 1;
         showValue = 1;
         numTicks => ((.max - .min) + 1);
         value<NEportLevels={2,2}> => <-.X_Slice;
      };
      UIdial Y_Dial<NEx=11.,NEy=165.> {
         parent => <-.Dial_Panel;
         y = 0;
         x => ((parent.width / 3) - 1);
         width => (parent.width / 3);
         height => (.width + 40);
         title => "Y Plane";
         value<NEportLevels={2,2}> => <-.Y_Slice;
         mode = 1;
         decimalPoints = 0;
         message = "Select y slice";
         min = 0.;
         max => (<-.Grid.dims[1] - 1);
         immediateMode = 1;
         showValue = 1;
         numTicks => ((.max - .min) + 1);
      };
      UIframe Widget_Frame<NEx=-231.,NEy=-44.> {
         parent => <-.UImod_panel;
         y => (<-.Load_File_Frame.height + <-.radio_box_frame.height);
         width => (parent.parent.width - 37);
         height = 190;
      };
      Field_Unif &Grid<NEx=-11.,NEy=-143.,NEportLevels={2,1}> => <-.PF_Grid_from_File.Hack;
      UIpanel Dial_Panel<NEx=-352.,NEy=110.> {
         visible => <-.Dials.set;
         parent => <-.Widget_Frame;
         y => (((parent.height - .height) / 2) - 10);
         height => <-.X_Dial.height;
         width => (parent.width - 3);
      };
      UIradioBox UIradioBox<NEx=-429.,NEy=-88.> {
         parent => <-.radio_box_frame;
         cmdList => {<-.Dials,
            <-.Sliders};
         selectedItem = 0;
         height => 40;
         y = 0;
         orientation = 1;
      };
      UIoption Dials<NEx=-429.,NEy=-55.> {
         do = 1;
         message = "Select to use dials";
         set = 1;
      };
      UIoption Sliders<NEx=-429.,NEy=-22.> {
         do = 1;
         message = "Select to use sliders";
      };
      UIframe radio_box_frame<NEx=-429.,NEy=-143.> {
         parent => <-.UImod_panel;
         y => <-.Load_File_Frame.height;
         width => (parent.parent.width - 37);
         height = 31;
      };
      UIslider X_Slider<NEx=-297.,NEy=242.> {
         parent => <-.Slider_Panel;
         y = 0;
         width => (parent.width - 10);
         message = "Select x slice";
         min = 0.;
         max => (<-.Grid.dims[0] - 1);
         mode = 1;
         immediateMode = 0;
         horizontal = 1;
         processingDirection = 0;
         increment = 1.;
         value<NEportLevels={2,2}> => <-.X_Slice;
         title => "X Plane";
      };
      UIslider Y_Slider<NEx=-154.,NEy=242.> {
         parent => <-.Slider_Panel;
         value<NEportLevels={2,2}> => <-.Y_Slice;
         y => <-.X_Slider.height;
         width => (parent.width - 10);
         message = "Select y slice";
         min = 0.;
         max => (<-.Grid.dims[1] - 1);
         mode = 1;
         immediateMode = 0;
         horizontal = 1;
         processingDirection = 0;
         increment = 1.;
         title => "Y Plane";
      };
      UIslider Z_Slider<NEx=-11.,NEy=242.> {
         parent => <-.Slider_Panel;
         value<NEportLevels={2,2}> => <-.Z_Slice;
         y => (<-.X_Slider.height + <-.Y_Slider.height);
         width => (parent.width - 10);
         message = "Select z slice";
         min = 0.;
         max => (<-.Grid.dims[2] - 1);
         mode = 1;
         immediateMode = 0;
         horizontal = 1;
         processingDirection = 0;
         increment = 1.;
         title => "Z Plane";
      };
      UIpanel Slider_Panel<NEx=-429.,NEy=176.> {
         visible => <-.Sliders.set;
         parent => <-.Widget_Frame;
         y = 0;
         width => parent.width;
      };
      int X_Slice<NEportLevels={0,2},NEx=-66.,NEy=-22.> = 0;
      int Y_Slice<NEportLevels={0,2},NEx=-66.,NEy=11.> = 0;
      int Z_Slice<NEportLevels={0,2},NEx=-66.,NEy=44.> = 0;
      UIdial Z_dial<NEx=154.,NEy=165.> {
         parent => <-.Dial_Panel;
         y = 0;
         x => (((parent.width / 3) * 2) - 2);
         width => (parent.width / 3);
         height => (.width + 40);
         value<NEportLevels={2,2}> => <-.Z_Slice;
         title => "Z Plane";
         mode = 1;
         decimalPoints = 0;
         message = "Select z slice";
         min = 0.;
         max => (<-.Grid.dims[2] - 1);
         immediateMode = 1;
         showValue = 1;
         numTicks => ((.max - .min) + 1);
      };
   };
   ParFlow.PF_Grid_from_File PF_Grid_from_File<NEx=495.,NEy=275.> {
      read_parflow {
         choose_file {
            input_file => <-.<-.<-.Slice3_UI.Load_File_Frame.Filename_UItext.text;
         };
      };
   };
};
