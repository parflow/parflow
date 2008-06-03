"$XP_PATH<0>/v/templ.v" Templates {
   WORKSPACE_1<process="user"> {
      macro read_parflow<NEx=352.,NEy=55.,NExOffset=200.,NEyOffset=64.,NEdisplayMode="open"> {
         module choose_file<NEx=0.,NEy=11.,build_dir="read_parflow",src_file="Choose.c",c_src_files="Choose.c",process="user"> {
            omethod+notify_inst+req choose_file_type<link_files=""> = "choose_type";
            string+read+notify input_file<NEportLevels={3,0},NEx=242.,NEy=55.,export=2>;
            int+write choice<NEportLevels={0,3}> = 0;
            string+write pfsb_file<NEportLevels={0,2},export=2,NEx=253.,NEy=242.>;
            string+write pfb_file<NEportLevels={0,2},NEx=253.,NEy=272.>;
         };
         module ReadPFSB<NEx=-33.,NEy=132.,process="user",src_file="ReadPFSB.c",c_src_files="ReadPFSB.c",build_dir="read_parflow"> {
            omethod+req ReadFile = "ReadPFSB";
            string+read+notify+req filename<NEportLevels={2,1},export=2> => <-.choose_file.pfsb_file;
            Field_Struct+write NonUniform_Grid<NEportLevels={0,3},export=2> {
               int orig_dims<NEportLevels={0,0},NEx=308.,NEy=66.>[3];
            };
         };
         module ReadPFB<NEx=132.,NEy=132.,build_dir="read_parflow",src_file="ReadPFB.c",c_src_files="ReadPFB.c",process="user"> {
            omethod+req ReadFile = "ReadPFB";
            string+read+notify+req filename<NEportLevels={2,1},export=2> => <-.choose_file.pfb_file;
            Field_Unif+write Uniform_Grid<NEportLevels={0,3},export=2> {
               nspace = 3;
               ndim = 3;
               points => {
                  0,0,0,0,0,0
               };
               nnode_data = 1;
               node_data = {
                  {
                     veclen=1,,,,,
                  }};
            };
         };
      };
      module sparse_to_unif<NEx=165.,NEy=33.,src_file="sparse_to_unif.c",build_dir="utility",c_src_files="sparse_to_unif.c"> {
         omethod+req update = "sparse_to_unif";
         Field_Struct+read+notify+req &in<NEportLevels={2,0},link_files="">;
         float+read+notify zeroval<NEportLevels={2,1}> = 0.;
         Field_Unif+write out<NEportLevels={0,2},NEx=198.,NEy=198.>;
      };
      macro Grid_Scale<NEx=473.,NEy=88.,NExOffset=171.,NEyOffset=46.> {
         Mesh &Mesh<NEx=-110.,NEy=-22.,NEportLevels={2,1}>;
         float min<NEportLevels=1,NEx=-165.,NEy=231.,NEwidth=209.,NEheight=77.>[] => Mesh.coordinates.min_vec;
         float max<NEportLevels={1,1},NEx=-165.,NEy=264.,NEwidth=209.,NEheight=77.>[] => Mesh.coordinates.max_vec;
         float max_xy_range<NEportLevels=1,NEx=11.,NEy=308.,NEwidth=473.,NEheight=66.> => max_array({.x_range,.y_range});
         float z_range#1<NEportLevels={1,1},NEx=11.,NEy=264.,NEwidth=473.,NEheight=66.> => (max[2] - min[2]);
         float y_range<NEportLevels={1,1},NEx=11.,NEy=231.,NEwidth=473.,NEheight=66.> => (max[1] - min[1]);
         float x_range<NEportLevels={1,1},NEx=11.,NEy=198.,NEwidth=473.,NEheight=66.> => (max[0] - min[0]);
         UIdial Xdial<NEx=77.,NEy=132.> {
            parent => <-.UImod_panel;
            value = 1.;
            title => "X";
            y = 0;
            width => (parent.width / 3);
            height => (.width + 40);
            min = 0.001;
            mode = 2;
            decimalPoints = 3;
            valuePerRev = 1.;
            showValue = 1;
         };
         UIdial Ydial<NEx=220.,NEy=132.> {
            parent => <-.UImod_panel;
            y = 0;
            height => (.width + 40);
            width => (parent.width / 3);
            value<NEportLevels={2,2}> = 1.;
            title => "Y";
            x => ((parent.width / 3) - 1);
            min = 0.001;
            mode = 2;
            decimalPoints = 3;
            valuePerRev = 1.;
         };
         UIdial Zdial<NEx=363.,NEy=132.> {
            parent => <-.UImod_panel;
            y = 0;
            height => (.width + 40);
            width => (parent.width / 3);
            value<NEportLevels={2,2}> = 1.;
            title => "Z";
            x => (((2 * parent.width) / 3) - 2);
            min = 0.001;
            mode = 2;
            decimalPoints = 3;
            valuePerRev = 10.;
         };
         UImod_panel UImod_panel<NEx=99.,NEy=33.> {
            option {
               set = 1;
            };
            parent<NEportLevels={3,0}>;
            title => "Grid Scale";
            width => (parent.width - 36);
         };
         DVM.DVscale DVscale<NEx=231.,NEy=231.> {
            in => <-.Mesh;
            scale_x => <-.Xdial.value;
            scale_y => <-.Ydial.value;
            scale_z => <-.Zdial.value;
            out<NEportLevels={0,3}>;
         };
      };
   };
   WORKSPACE_2 {
      macro Load_File_Frame<NEx=121.,NEy=352.,NExOffset=91.,NEyOffset=45.> {
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
            x<export=3>;
            y<export=3> = 0;
            width<export=3> => (parent.width - 10);
            parent<NEportLevels={3,0}>;
            height<export=3> = 95;
            visible<NEportLevels={3,0}>;
         };
         UItext Filename_UItext<NEx=132.,NEy=55.> {
            x = 7;
            y = 10;
            width => (parent.width - 25);
            parent => <-.UIframe;
            height = 28;
            message = "Current File";
            text<NEportLevels={2,3},export=2> => <-.FileDialog.filename;
            rows = 1;
            resizeToText = 1;
            outputOnly = 0;
         };
      };
      macro Slice3_UI<NEx=330.,NEy=462.,NExOffset=462.,NEyOffset=232.> {
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
         Field_Unif &Grid<NEx=-11.,NEy=-143.,NEportLevels={2,1}>;
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
      macro PF_Grid_from_File<NEx=297.,NEy=88.,NExOffset=378.,NEyOffset=132.> {
         WORKSPACE_1.read_parflow read_parflow<NEx=-165.,NEy=-121.,export=1,NEdisplayMode="closed"> {
            choose_file {
               input_file<NEportLevels={4,0}>;
            };
            ReadPFB<export=2> {
               Uniform_Grid<export=3,NEdisplayMode="open"> {
                  coordinates {
                     min<NEdisplayMode="open">;
                  };
                  node_data<NEdisplayMode="open">;
               };
            };
         };
         macro scale_grid<NExOffset=-78.,NEyOffset=-11.,NEx=-143.,NEy=176.> {
            DVM.DVscale DVscale<NEx=176.,NEy=352.> {
               in => <-.link;
               scale_x => <-.Xdial.value;
               scale_y => <-.Ydial.value;
               scale_z => <-.Zdial.value;
               out<NEportLevels={0,3}>;
            };
            link link<NEportLevels={2,1},NEx=165.,NEy=57.> => <-.Field_Chooser.out;
            UImod_panel UImod_panel<NEx=352.,NEy=99.> {
               option {
                  set = 1;
               };
               title => "Grid Scale";
            };
            UIdial Xdial<NEx=374.,NEy=264.> {
               parent => <-.UImod_panel;
               value<NEportLevels={2,2}> => <-.copy_on_change.output;
               title => "X";
               y = 0;
               width => (parent.width / 3);
               height => (.width + 40);
               min = 0.001;
               mode = 2;
               decimalPoints = 3;
               valuePerRev = 1.;
               showValue = 1;
            };
            UIdial Ydial<NEx=517.,NEy=264.> {
               parent => <-.UImod_panel;
               y = 0;
               height => (.width + 40);
               width => (parent.width / 3);
               value<NEportLevels={2,2}> => <-.copy_on_change#1.output;
               title => "Y";
               x => ((parent.width / 3) - 1);
               min = 0.001;
               mode = 2;
               decimalPoints = 3;
               valuePerRev = 1.;
            };
            UIdial Zdial<NEx=660.,NEy=264.> {
               parent => <-.UImod_panel;
               y = 0;
               height => (.width + 40);
               width => (parent.width / 3);
               value<NEportLevels={2,2}> => <-.copy_on_change#2.output;
               title => "Z";
               x => (((2 * parent.width) / 3) - 2);
               min = 0.001;
               mode = 2;
               decimalPoints = 3;
               valuePerRev = 10.;
            };
            GMOD.copy_on_change copy_on_change<NEx=352.,NEy=198.> {
               trigger => <-.UIbutton.do;
               input = 1;
               on_inst = 0;
               output = 1;
            };
            GMOD.copy_on_change copy_on_change#1<NEx=495.,NEy=198.> {
               trigger => <-.UIbutton.do;
               input = 1;
               output = 1.;
               on_inst = 0;
            };
            GMOD.copy_on_change copy_on_change#2<NEx=660.,NEy=198.> {
               trigger => <-.UIbutton.do;
               input = 1;
               output = 1.;
               on_inst = 0;
            };
            UIbutton UIbutton<NEx=561.,NEy=55.> {
               parent => <-.UImod_panel;
               label => "Reset";
               y => (<-.Xdial.height + 15);
               x = 10;
               width = 55;
            };
         };
         group Field_Chooser<NEx=-132.,NEy=77.> {
            int choice<NEportLevels={2,1}> => <-.read_parflow.choose_file.choice;
            Field_Unif &objects<NEx=605.,NEy=264.,NEportLevels={2,0}>[] => {
               <-.sparse_to_unif.out,<-.read_parflow.ReadPFB.Uniform_Grid};
            Field_Unif &out<NEportLevels={0,2},NEx=198.,NEy=440.> => objects[(!(!.choice))];
         };
         Field_Struct &Hack<NEx=-66.,NEy=253.,NEportLevels={1,2}> => .scale_grid.DVscale.out;
         WORKSPACE_1.sparse_to_unif sparse_to_unif<NEx=-33.,NEy=-33.> {
            in => <-.read_parflow.ReadPFSB.NonUniform_Grid;
         };
      };
   };
   WORKSPACE_3<process="user"> {
      group Field_Chooser<NEx=440.,NEy=352.> {
         int choice<NEportLevels={2,1}>;
         Field_Unif &objects<NEx=605.,NEy=264.,NEportLevels={2,0}>[];
         Field_Unif &out<NEportLevels={0,2},NEx=198.,NEy=440.,NEdisplayMode="open"> => objects[(!(!choice))];
      };
      macro FileDialog_button<NEx=55.,NEy=132.,NExOffset=330.,NEyOffset=185.> {
         UIshell UIshell<NEx=-297.,NEy=-99.> {
            visible<NEportLevels={3,0}>;
            parent<NEportLevels={3,0}>;
            width = 187;
            height = 192;
         };
         UIbutton UIbutton<NEx=-297.,NEy=-55.> {
            parent => <-.UIshell;
            label => "Popup Dialog";
            x => ((parent.width - .width) / 2);
            y => ((parent.height - .height) / 2);
         };
         UIfileDialog UIfileDialog<NEx=-297.,NEy=-11.> {
            visible => <-.UIbutton.do;
            filename<NEportLevels={0,3}>;
         };
      };
      macro DirDialog_button<NEx=66.,NEy=132.,NExOffset=330.,NEyOffset=185.> {
         UIshell UIshell<NEx=-297.,NEy=-99.> {
            x = 272;
            y = 796;
            width = 123;
            height = 63;
            visible<NEportLevels={3,0}>;
            parent<NEportLevels={3,0}>;
         };
         UIbutton UIbutton<NEx=-297.,NEy=-55.> {
            x => ((parent.width - .width) / 2);
            parent => <-.UIshell;
            y => ((parent.height - .height) / 2);
            label => "Popup Dialog";
         };
         UIdirectoryDialog UIdirectoryDialog<NEx=-297.,NEy=-11.> {
            visible => <-.UIbutton.do;
            filename<NEportLevels={0,3}>;
         };
      };
      library string_functions<NEeditable=1,NEreparentable=1,NEx=22.,NEy=0.,process="user"> {
         module strcat<NEx=253.,NEy=99.,src_file="string.c",build_dir="utility",c_src_files="string.c",process="user"> {
            omethod+notify_inst+req update = "string_concat";
            string+read+notify string1<NEportLevels={2,0},export=2>;
            string+read+notify string2<NEportLevels={2,0},export=2>;
            string+write concat<NEportLevels={0,2}>;
         };
         module strsplit_fwd<NEx=154.,NEy=110.,process="user",src_file="string.c",c_src_files="string.c",build_dir="utility"> {
            omethod+notify_inst+req update = "string_split_fwd";
            string+read+notify string1<NEportLevels={2,0},build_dir="",export=2>;
            string+read+notify string2<NEportLevels={2,0},export=2>;
            string+write outstring1<NEportLevels={0,2}>;
            string+write outstring2<NEportLevels={0,2}>;
         };
         module strsplit_rev<NEx=154.,NEy=143.,process="user",src_file="string.c",c_src_files="string.c",build_dir="utility"> {
            omethod+notify_inst+req update = "string_split_rev";
            string+read+notify string1<NEportLevels={2,0},build_dir="",export=2>;
            string+read+notify string2<NEportLevels={2,0},export=2>;
            string+write outstring1<NEportLevels={0,2}>;
            string+write outstring2<NEportLevels={0,2}>;
         };
         module strsplit_all<NEx=154.,NEy=176.,process="user",src_file="string.c",build_dir="utility",c_src_files="string.c"> {
            omethod+notify_inst+req update = "string_split_all";
            string+read+notify string1<NEportLevels={2,0}>;
            string+read+notify string2<NEportLevels={2,0}>;
            int strings<NEportLevels={0,2}>;
            string+write array<NEportLevels={0,2}>[.strings];
         };
      };
      macro pfsb_series_basename<NEx=550.,NEy=33.,NExOffset=304.,NEyOffset=213.> {
         string dot<NEportLevels={1,1},NEx=-110.,NEy=-143.> = ".";
         strsplit_rev strsplit_rev<NEx=55.,NEy=-44.> {
            string1<NEportLevels={3,0}>;
            string2 => <-.dot;
         };
         strsplit_rev strsplit_rev#1<NEx=0.,NEy=22.> {
            string1<NEportLevels={3,0}> => <-.strsplit_rev.outstring1;
            string2 => <-.dot;
            outstring1<NEportLevels={0,3}>;
         };
      };
   };
   NELIBS {
      UTILITY {
         Visualization<NEdisplayMode="open">;
      };
   };
   FILE {
      file_find_expr {
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         string *pattern<NEportLevels={2,0}>;
      };
      file_array_ascii {
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
         file *file;
         int *offset<NEportLevels={2,0}>;
         int *type<NEportLevels={2,0}>;
         int *columns<NEportLevels={2,0}>[];
         boolean *ascii_binary<NEportLevels={2,0}>;
      };
   };
   flibrary+file_group USER<NEeditable=1,process="user",libdeps="FLD",link_files="-luser",build_dir="user",build_cmd="$(MAKE)"> {
      module Mesh_plot_2D_xform<NEx=121.,NEy=11.,src_file="2Dxform.c",build_dir="utility",c_src_files="2Dxform.c",process="user"> {
         omethod+req update = "calculate";
         int+read+notify+req axis<NEportLevels={2,0}>;
         Mesh+read+notify+req &in<NEportLevels={2,0}>;
         Xform+write Xform;
         Mesh &out<NEportLevels={0,2}> => merge(Xform,in);
      };
      module EVselect_tester<NEx=165.,NEy=121.,build_dir="utility",src_file="test.c",c_src_files="test.c",link_files="-lnsl -lsocket",process="user"> {
         omethod+notify_inst+req update = "update";
         omethod+notify_inst omethod_inst = "constructor";
      };
      module client_tester<NEx=121.,NEy=77.,build_dir="utility",src_file="client_tester2.c",c_src_files="client_tester2.c",link_files="-lnsl -lsocket",process="user",NEdisplayMode="open"> {
         omethod+notify_inst+req initialize<weight=0> = "connect_socket";
         omethod+notify_inst update = "get_menu";
         omethod+notify_deinst cleanup<weight=2> = "disconnect_socket";
         UIshell UIshell<NEx=440.,NEy=165.>;
         int+notify notify<NEportLevels={1,1},NEx=440.,NEy=195.> => .UpdateMenu_button.do;
         string component<NEportLevels={0,1},NEx=440.,NEy=225.>[] = {""};
         UIoptionBoxLabel UIoptionBoxLabel<NEx=440.,NEy=255.> {
            parent => <-.UIshell;
            labels => <-.component;
            UIlabel {
               label => "Select Components";
            };
         };
         UIbutton UpdateMenu_button {
            parent => <-.UIshell;
            label => "Update Menu";
            x = 15;
            y => (<-.UIoptionBoxLabel.height + 15);
         };
         UIbutton ReadComponents_button<NEx=374.,NEy=253.,NEdisplayMode="open"> {
            parent => <-.UIshell;
            label => "Read Components";
            x => ((<-.UpdateMenu_button.width + <-.UpdateMenu_button.x) + 15);
            y => <-.UpdateMenu_button.y;
            width = 130;
         };
      };
      module test_copy<NEx=176.,NEy=33.,src_file="test_copy.c",build_dir="utility",c_src_files="test_copy.c"> {
         omethod+notify_inst+req update = "test_copy";
         group group;
         group &group_ref<NEportLevels={1,0}> => .group;
      };
   };
   EXAMPLES {
      Applications<NEdisplayMode="open">;
   };
   WORKSPACE_XP<process="user"> {
      module copy_mat<NEx=22.,NEy=22.,build_dir="utility",src_file="misc.c",c_src_files="misc.c",process="user"> {
         omethod+req update = "copy_mat";
         float+read+notify+req in<NEportLevels={2,0}>[4][4];
         float+write out<NEportLevels={0,2}>[4][4];
      };
      macro loop_pfsb_files<NEx=55.,NEy=341.,NExOffset=135.,NEyOffset=35.> {
         GMOD.shell_command shell_command<NEx=77.,NEy=143.,process="express"> {
            command => (("ls " + <-.basename) + ".*.pfsb");
         };
         WORKSPACE_3.strsplit_all split_lines<NEx=77.,NEy=187.> {
            string1 => <-.shell_command.stdout_string;
            string2 = "\n";
         };
         prim prim<NEportLevels={1,1},NEx=286.,NEy=110.>[] => .split_lines.array;
         DVM.DVswitch DVswitch<NEx=429.,NEy=143.,process="express"> {
            in => <-.prim;
            index => <-.loop.count;
         };
         WORKSPACE_3.pfsb_series_basename pfsb_series_basename<NEx=-121.,NEy=66.> {
            strsplit_rev {
               string1<NEportLevels={4,0}>;
            };
         };
         GMOD.loop loop<NEx=-77.,NEy=11.,process="express"> {
            reset<NEportLevels={3,2}> => ;
            run<NEportLevels={3,0}> => ;
            cycle<NEportLevels={3,0}>;
            done = 1;
            start_val = 0.;
            end_val => (<-.split_lines.strings - 2);
            incr = 1.;
            count = 5.;
         };
         string current_file<NEportLevels={1,2},NEx=-99.,NEy=308.,NEwidth=616.,NEheight=66.>;
         string basename<NEportLevels=1,NEx=-121.,NEy=121.> => .pfsb_series_basename.strsplit_rev#1.outstring1;
      };
   };
};
