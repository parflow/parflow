#
# evap.tcl - evaluate_parameters 2.1 for Tcl
#
# lusol@lehigh.EDU, 94/05/14
#
# Made to conform, as much as possible, to the C function evap. The C, Perl
# and Tcl versions of evap are patterned after the Control Data procedure
# CLP$EVALUATE_PARAMETERS for the NOS/VE operating system, although neither
# approaches the richness of CDC's implementation.
#
# Availability is via anonymous FTP from ftp.Lehigh.EDU (128.180.63.4) in the
# directory pub/evap/evap-2.x.
#
# Stephen O. Lidie, Lehigh University Computing Center.
#
# Copyright (C) 1994 - 1994 by Stephen O. Lidie and Lehigh University.
# All rights reserved.
#
# For related information see the evap/C header file evap.h.  Complete
# help can be found in the man pages evap(2), evap.c(2), evap.pl(2),
# evap.tcl(2) and evap_pac(2).
#     
# 
#                           Revision History 
#
# lusol@Lehigh.EDU 94/03/14 (PDT version 2.0)  Version 2.1
#   . Original release - similar to version 2.1 of the C and Perl functions
#     evap.
#
# $Id: evap.tcl,v 1.1 1995/06/30 00:48:21 ssmith Exp $
#


proc evap { PDT MM {QUAL ""} } {

    global evap_QUAL evap_message_modules
    set evap_QUAL $QUAL
    set evap_message_modules "./libevapmm.a"
    global argc argv0 argv env ${evap_QUAL}options ${evap_QUAL}Options evap_embed
    if { ! [info exists evap_embed] } { set evap_embed 0 }
    global evap_PARAMETER evap_INFO evap_ALIAS evap_REQUIRED evap_VALID_VALUES evap_ENV evap_EVALUATE evap_DEFAULT_VALUE

    if { $evap_embed } {
	catch {unset ${evap_QUAL}opt_help}
	catch {unset ${evap_QUAL}Options(help)}
	catch {unset ${evap_QUAL}options(help)}
	unset evap_PARAMETER
	unset evap_INFO
	unset evap_ALIAS
	unset evap_REQUIRED
	unset evap_VALID_VALUES
	unset evap_ENV
	unset evap_EVALUATE
	unset evap_DEFAULT_VALUE
    }

    set evap_PARAMETER {};                                               # no parameter names
    set evap_INFO() {};                                                  # no encoded parameter information
    set evap_ALIAS() {};                                                 # no aliases
    set evap_REQUIRED {};                                                # no required parameters
    set evap_VALID_VALUES() {};                                          # no keywords
    set evap_ENV() {};                                                   # no default environment variables
    set evap_EVALUATE() {};                                              # no PDT values evaluated yet
    set evap_DEFAULT_VALUE() {};                                         # no default values yet

    global evap_DOS;             set evap_DOS 0;                         # 1 IFF MS-DOS beast
    global evap_error;           set evap_error 0;                       # no PDT parsing or command line errors 
    global evap_full_help;       set evap_full_help 0;                   # full_help flag
    global evap_usage_help;      set evap_usage_help 0;                  # usage_help flag
    global evap_file_list;       set evap_file_list {optional_file_list};# file_list flag

    global evap_d;               set evap_d "\[0-9\]";                   # simulate Perl's \d, match digit
    global evap_D;               set evap_D "\[^0-9\]";                  # simulate Perl's \D, match non-digit
    global evap_s;               set evap_s "\[ \t\n\r\f\]";             # simulate Perl's \s, match whitespace
    global evap_S;               set evap_S "\[^ \t\n\r\f\]";            # simulate Perl's \S, match non-whitespace
    global evap_w;               set evap_w "\[0-9a-z_A-Z]";             # simulate Perl's \w, match word
    global evap_W;               set evap_W "\[^0-9a-z_A-Z]";            # simulate Perl's \W, match non-word

    global evap_pdt_reg_exp1;    set evap_pdt_reg_exp1 {^(.)(.)(.?)$}
    global evap_pdt_reg_exp2;    set evap_pdt_reg_exp2 {^TRUE$|^YES$|^ON$|^1$}
    global evap_pdt_reg_exp3;    set evap_pdt_reg_exp3 {^FALSE$|^NO$|^OFF$|^0$}
    global evap_pdt_reg_exp4;    set evap_pdt_reg_exp4 "^$evap_s*no_file_list$evap_s*\$"
    global evap_pdt_reg_exp5;    set evap_pdt_reg_exp5 "^$evap_s*optional_file_list$evap_s*\$"
    global evap_pdt_reg_exp6;    set evap_pdt_reg_exp6 "^$evap_s*required_file_list$evap_s*\$"
    global evap_pdt_reg_exp7;    set evap_pdt_reg_exp7 "^$evap_s*list$evap_s+of$evap_s+"

    set local_pdt $PDT
    set local_pdt [linsert $PDT 0 {help, ?: switch}];                # supply -help automatically

    foreach option $local_pdt {

	if { [regexp -nocase "^#.*|$evap_s*PDT$evap_s+|PDT\$" $option] } { continue }
	regsub -nocase "^$evap_s*PDTEND" $option {} option
	if { [regexp {^ ?$} $option] } { continue }

	if { [regexp "$evap_pdt_reg_exp4|$evap_pdt_reg_exp5|$evap_pdt_reg_exp6" $option] } {
	    set evap_file_list $option
	    continue
	}

	catch {unset parameter alias rest}
	regexp "^$evap_s*($evap_S*)$evap_s*,$evap_s*($evap_S*)$evap_s*:$evap_s*(.*)$" $option ignore parameter alias rest;
	if { ! [info exists parameter] || ! [info exists alias] || ! [info exists rest] } {
	    EvaP_PDT_error "Error in an evaluate_parameters 'parameter, alias: type' option specification:  \"$option\"."
	    continue
	}
	if { [info exists evap_INFO($parameter)] } {
	    EvaP_PDT_error "Duplicate parameter $parameter: \"$option\"."
	    continue
	}
	lappend evap_PARAMETER $parameter

	catch {unset type}
	regexp "($evap_pdt_reg_exp7)*$evap_s*(switch|integer|string|real|file|boolean|key|name|application)$evap_s*(.*)" \
	    $rest ignore list type rest
	if { ! [info exists type] } {
	    EvaP_PDT_error "Parameter $parameter has an undefined type:  \"$option\"."
	    continue
	}
	if { $list != "" && ! [regexp "$evap_pdt_reg_exp7" $list] } {
	    EvaP_PDT_error "Expecting 'list of', found:  \"$list\"."
	    continue
	}
	set list [expr { ($list != "") ? 1 : 0 }]
	if { $type == {switch} } { set type {w} }
	set type [string range $type 0 0]

	set default_value {}
	regexp "$evap_s*=$evap_s*(.*)" $rest ignore default_value
	if { [regexp "^\(\[^\(\]\)($evap_w*)$evap_s*,$evap_s*(.*)" $default_value ignore p1 p2 p3] } {
	    set default_value $p3
	    set evap_ENV($parameter) "$p1$p2"
	}

	set required [expr { ($default_value == {$required}) ? "R" : "O" }]
	set evap_INFO($parameter) "$required$type$list"
	if { $required == "R" } { lappend evap_REQUIRED $parameter }

	if { $type == "k" } {
	    regsub {keyend.*} $rest {} rest
	    regsub -all {,} $rest {} rest
	    set rest [string trim $rest]
	    regsub -all { +} $rest { } evap_VALID_VALUES($parameter)
	    
	}

	set alias_error 0
	foreach value [array name evap_ALIAS] {
	    if { $alias == $evap_ALIAS($value) } {
		EvaP_PDT_error "Duplicate alias $alias:  \"$option\"."
		set alias_error 1
	    }
	}
	if { $alias_error } { continue }
	set evap_ALIAS($parameter) $alias

	if { [regexp {^.w1$} $evap_INFO($parameter)] } {
	    EvaP_PDT_error "Cannot have 'list of' switch:  \"$option\"."
	    continue
	}

	if { $default_value != ""  && $default_value != {$required} } {
	    if { [info exists env(evap_ENV($parameter))] } {
		if { $env($evap_ENV($parameter)) != "" } {
		    set default_value $env($evap_ENV($parameter))
		}
	    }
	    set evap_DEFAULT_VALUE($parameter) $default_value
	    EvaP_set_value 0 $type $list $default_value $parameter
	}

    }; # forend options

    if { $evap_error } {
	puts stderr "Inspect the file \"evap.tcl\" for details on PDT syntax."
	exit 1
    }

    while { $argc > 0 } {

	incr argc -1
	set option [lindex $argv 0]
	set argv [lreplace $argv 0 0]
	catch {unset value}

	if { [regexp {^-(full_help|\?\?\?)$} $option] } { set evap_full_help 1 }
	if { [regexp {^-(usage_help|\?\?)$}  $option] } { set evap_usage_help 1 }
	if { $evap_full_help || $evap_usage_help } { set option {-help} }
	
	if { [regexp {^(--|-)(.*)} $option p1 p2 p3] } {
	    if { $option == {--} } {
		return [EvaP_fin $MM]
	    }
	    set option $p3
	} else {
	    set argv [linsert $argv 0 $option]
	    incr argc 1
	    return [EvaP_fin $MM]
	}

	foreach alias [array names evap_ALIAS] {
	    if { $option == $evap_ALIAS($alias) } { set option $alias }
	}

	if { ! [info exists evap_INFO($option)] } {
	    set found 0
	    set length [string length $option]
	    foreach key [array names evap_INFO] {
		if { $option == [string range $evap_INFO($key) 0 [expr $length - 1]] } {
		    if { $found } {
			puts stderr "Ambiguous parameter: -$option."
			incr evap_error 1
			break; # substring search
		    }
		    set found $key
		}
	    }; # forend substring search for parameter
	    set option [expr { ($found != 0) ? $found : $option } ]
	    if { ! [info exists evap_INFO($option)] } {
		puts stderr "Invalid parameter: -$option."
		incr evap_error 1
		continue; # arguments
	    }
	}; # ifend substring search

	regexp "$evap_pdt_reg_exp1" $evap_INFO($option) ignore required type list

	if { $type != {w} } {
	    if { $argc <= 0 } {
		puts stderr "Value required for parameter -$option."
		incr evap_error 1
		continue; # arguments
	    } else {
		set value [lindex $argv 0]
		set argv [lreplace $argv 0 0]
		incr argc -1
	    }
	}

	switch -exact $type {

	    {w} {; # switch
		set value 1
	    }

	    {i} {; # integer
		if { ! [regexp {^[+-]?[0-9]+$} $value] } {
		    puts stderr "Expecting integer reference, found \"$value\" for parameter -$option."
		    incr evap_error 1
		    unset value
		}
	    }

	    {r} {; # real
		if { ! [regexp "^$evap_s*\[+-\]?($evap_d+(\.$evap_d*)?|\.$evap_d+)(\[eE\]\[+-\]?$evap_d+)?$evap_s*\$" $value] } {
		    puts stderr "Expecting real reference, found \"$value\" for parameter -$option."
		    incr evap_error 1
		    unset value
		}
	    }

	    {s} {; # string
	    }

	    {n} {; # name
	    }

	    {a} {; # application
	    }

	    {f} {; # file
		if { [string length $value] > 255 } {
		    puts stderr "Expecting file reference, found \"$value\" for parameter -$option."
		    incr evap_error 1
		    unset value
		}
	    }

	    {b} { ; # boolean
		if { ! [regexp -nocase "$evap_pdt_reg_exp2|$evap_pdt_reg_exp3" $value] } {
		    puts stderr "Expecting boolean, found \"$value\" for parameter -$option."
		    incr evap_error 1
		    unset value
		}
	    }

	    {k} {; # keyword - first try an exact match, then a substring match
		catch {unset found}
		set keys [split $evap_VALID_VALUES($option) { }]
		for {set i 0} {$i < [llength $keys] && ! [info exists found]} {incr i 1} {
		    set key [lindex $keys $i]
		    if { $value == $key } { set found 1 }
		}
		if { ! [info exists found] } {
		    set length [string length $value]
		    for {set i 0} {$i < [llength $keys]} {incr i 1} {
			set key [lindex $keys $i]
			if { $value == [string range $key 0 [expr $length - 1]] } {
			    if { [info exists found] } {
				puts stderr "Ambiguous keyword for parameter -$option: $value."
				incr evap_error 1
				break
			    }
			    set found $key
			}
		    }; # forend
		    set value [expr { ([info exists found]) ? $found : $value } ]
		}; # ifend
		if { ! [info exists found] } {
		    puts stderr "\"$value\" is not a valid value for the parameter -$option."
		    incr evap_error 1
		    unset value
		}
	    }

	}; # switchend

	if { ! [info exists value] } { continue }

	if { $list == 1 } { set list 2 }
	EvaP_set_value 1 $type $list $value $option
	set index [lsearch $evap_REQUIRED $option]
	if { $index != -1 } { set evap_REQUIRED [lreplace $evap_REQUIRED $index $index] }
	if { $list } { set evap_INFO($option) "$required${type}3" }

    }; # whilend arguments

    return [EvaP_fin $MM]
	
}; # end evap




proc EvaP_fin {MM} {

    #
    # Finish up evaluate_parameters processing:
    #
    # If -usage_help, -help or -full_help was requested then do it and exit.  Else,
    #
    #  . Store program name in help variables.
    #  . Perform deferred evaluations.
    #  . Ensure all $required parameters have been given a value.
    #  . Ensure the validity of the trailing file list.
    #  . Exit with a Unix return code of 1 if there were errors and $evap_embed = 0,
    #    else return to the calling Tcl program with a proper return code.
    #

    global evap_QUAL
    global evap_embed ${evap_QUAL}options ${evap_QUAL}Options ${evap_QUAL}opt_help
    global evap_error evap_full_help evap_usage_help argv0 evap_PARAMETER evap_INFO evap_ALIAS evap_pdt_reg_exp5
    global evap_d evap_D evap_s evap_S evap_w evap_W evap_pdt_reg_exp4 evap_pdt_reg_exp3 env evap_Help_Hooks evap_pdt_reg_exp6
    global evap_file_list evap_pdt_reg_exp1 evap_VALID_VALUES evap_ENV evap_EVALUATE evap_DEFAULT_VALUE evap_REQUIRED argc

    # Define Help Hooks text as required.

    if { ! [info exists evap_Help_Hooks(P_HHURFL)] } { set evap_Help_Hooks(P_HHURFL) " file(s)\n" }
    if { ! [info exists evap_Help_Hooks(P_HHUOFL)] } { set evap_Help_Hooks(P_HHUOFL) " \[file(s)\]\n" }
    if { ! [info exists evap_Help_Hooks(P_HHUNFL)] } { set evap_Help_Hooks(P_HHUNFL) "\n" }
    if { ! [info exists evap_Help_Hooks(P_HHBRFL)] } { set evap_Help_Hooks(P_HHBRFL) \
                                                            "\nfile(s) required by this command\n\n" }
    if { ! [info exists evap_Help_Hooks(P_HHBOFL)] } { set evap_Help_Hooks(P_HHBOFL) \
                                                            "\n\[file(s)\] optionally required by this command\n\n" }
    if { ! [info exists evap_Help_Hooks(P_HHBNFL)] } { set evap_Help_Hooks(P_HHBNFL) "\n" }
    if { ! [info exists evap_Help_Hooks(P_HHERFL)] } { set evap_Help_Hooks(P_HHERFL) \
                                                            "Trailing file name(s) required.\n" }
    if { ! [info exists evap_Help_Hooks(P_HHENFL)] } { set evap_Help_Hooks(P_HHENFL) \
                                                            "Trailing file name(s) not permitted.\n" }

    if { [info exists ${evap_QUAL}Options(help)] } {

	set type_list(w) switch
	set type_list(i) integer
	set type_list(s) string
	set type_list(r) real
	set type_list(f) file
	set type_list(b) boolean
	set type_list(k) key
	set type_list(n) name
	set type_list(a) application

	# Establish the proper pager and open the pipeline.  Do no paging
	# if the boolean environment variable D_EVAP_DO_PAGE is FALSE.

	set PAGER "more"
	if { [info exists env(PAGER)] } {
	    if { $env(PAGER) != "" } { set PAGER "$env(PAGER)" }
	}
	if { [info exists env(MANPAGER)] } {
	    if { $env(MANPAGER) != "" } { set PAGER "$env(MANPAGER)" }
	}
	set PAGER "|$PAGER"
	if { [info exists env(D_EVAP_DO_PAGE)] } {
	    if { $env(D_EVAP_DO_PAGE) != "" } {
		if { [regexp -nocase "$evap_pdt_reg_exp3" $env(D_EVAP_DO_PAGE)] } {
		    set PAGER "stdout"
		}
	    }
	}
	if { $PAGER != "stdout" } { set PAGER [open "$PAGER" "w"] }

	if { $evap_full_help } { puts -nonewline $PAGER "Command Source:  $argv0\n\n\n\n" }
	
	# Print the Message Module text and save any full help.  The key
	# is the parameter name and the value is a list of strings with
	# the newline as a separator.  If there is no Message Module or
	# it's empty then display an abbreviated usage message.

	if { $evap_usage_help || ! [info exists MM] || [llength $MM] <= 0 } {

	    set basename [split $argv0 /]; # only basename for usage help
	    puts -nonewline $PAGER "\nUsage: [lindex $basename [expr [llength $basename] - 1]]"
	    set optional {}
	    foreach p $evap_PARAMETER {
		if { [regexp {^R..?$} $evap_INFO($p)] } {; # if $required
		    puts -nonewline $PAGER " -$evap_ALIAS($p)";
		} else {
		    append optional " -$evap_ALIAS($p)"
		}
	    }
	    if { $optional != {} } { puts -nonewline $PAGER " \[$optional\]" }
	    if { [regexp "$evap_pdt_reg_exp5" $evap_file_list] } {
	        puts -nonewline $PAGER "$evap_Help_Hooks(P_HHUOFL)"
	    } elseif { [regexp "$evap_pdt_reg_exp6" $evap_file_list] } {
		puts -nonewline $PAGER "$evap_Help_Hooks(P_HHURFL)"
	    } else {
		puts -nonewline $PAGER "$evap_Help_Hooks(P_HHUNFL)"
	    }

	} else {

	    set parameter_help_in_progress 0
	    foreach m $MM {
		if { [regexp {^\.(.*)$} $m ignore p] } {; # look for `dot' leading character
		    set parameter_help_in_progress 1
		    set parameter_help($p) "\n"
		    continue
		}
		if { $parameter_help_in_progress } {
		    append parameter_help($p) "$m\n"
		} else {
		    puts -nonewline $PAGER "$m\n"
		}
	    }; # forend all lines in Message Module

	}; # ifend usage_help

	# Pass through the PDT list printing a standard EvaP help summary.

	puts -nonewline $PAGER "\nParameters:\n"
	if { ! $evap_full_help } { puts -nonewline $PAGER "\n" }

	foreach p $evap_PARAMETER {

	    if { $evap_full_help } { puts -nonewline $PAGER "\n" }
	    
	    if { $p == {help} } {
		puts -nonewline $PAGER "-$p, $evap_ALIAS($p), usage_help, full_help: Display Command Information\n"
		if { $evap_full_help } {
		    puts -nonewline $PAGER "\n\tDisplay information about this command, which includes\n"
		    puts -nonewline $PAGER "\ta command description with examples, plus a synopsis of\n"
		    puts -nonewline $PAGER "\tthe command line parameters.  If you specify -full_help\n"
		    puts -nonewline $PAGER "\trather than -help complete parameter help is displayed\n"
		    puts -nonewline $PAGER "\tif it's available.\n"
		}
		continue; # foreach parameter
	    }

	    regexp "$evap_pdt_reg_exp1" $evap_INFO($p) ignore required type list
	    set type $type_list($type)
	    set is_string [expr { ($type == {string}) ? 1 : 0 }]

	    puts -nonewline $PAGER "-$p, $evap_ALIAS($p): [expr { $list ? "list of " : "" }]$type"

	    if { $type == {key} } {
		puts -nonewline $PAGER " [join [split $evap_VALID_VALUES($p) " "] ", "], keyend"
	    }

	    global ${evap_QUAL}opt_$p
	    set def [info exists ${evap_QUAL}opt_$p]
	    
	    if { [regexp {^O$} $required] || $def } {; # if $optional or defined
		
		if { ! $def } {; # undefined and $optional
		    puts -nonewline $PAGER "\n"
		} else {; # defined (either $optional or $required), display default value(s)
		    set value [set ${evap_QUAL}opt_$p]
		    if { $list } {
			puts -nonewline $PAGER "[expr { [info exists evap_ENV($p)] ? " = $evap_ENV($p), " : " = " }]"
			puts -nonewline $PAGER "[expr { $is_string ? "(\"" : "(" }][ \
                                            expr { $is_string ? [join $value "\", \""] : [join $value ", "] }][ \
                                            expr { $is_string ? "\")\n" : ")\n" }]"
		    } else {
			puts -nonewline $PAGER "[expr { [info exists evap_ENV($p)] ? " = $evap_ENV($p), " : " = " }]"
			puts -nonewline $PAGER "[expr { $is_string ? "\"" : "" }]$value[expr { $is_string ? "\"\n" : "\n" }]"
		    }
		}
	    } elseif { [regexp {^R$} $required] } {; # if $required
		puts -nonewline $PAGER "[expr { [info exists evap_ENV($p)] ? " = $evap_ENV($p), " : " = " }]"
		puts -nonewline $PAGER "\$required\n"
	    } else {
		puts -nonewline $PAGER "\n"
	    }; # ifend $optional or defined parameter

	    if { $evap_full_help } {
		if { [info exists parameter_help($p)] } {
		    puts -nonewline $PAGER "$parameter_help($p)"
		} else {
		    puts -nonewline $PAGER "\n"
		}
	    }

	}; # forend all parameters

	if { [regexp "$evap_pdt_reg_exp5" $evap_file_list] } {
		puts -nonewline $PAGER "$evap_Help_Hooks(P_HHBOFL)"
	} elseif { [regexp "$evap_pdt_reg_exp6" $evap_file_list] } {
		puts -nonewline $PAGER "$evap_Help_Hooks(P_HHBRFL)"
	} else {
		puts -nonewline $PAGER "$evap_Help_Hooks(P_HHBNFL)"
	}

	if { $PAGER != "stdout" } {
	    close $PAGER; # flush the pipeline (required!)
	}
	if { $evap_embed } {
	    return -1;
	} else {
	    exit 0
	}

    }; # ifend help requested

    # Evaluate remaining unspecified command line parameters.  This has been deferred to now so that
    # if -help was requested the user sees unevaluated boolean, file and bacticked values.

    foreach parameter $evap_PARAMETER {
	if { ! [info exists evap_EVALUATE($parameter)] && [info exists evap_DEFAULT_VALUE($parameter)] } {
	    regexp "$evap_pdt_reg_exp1" $evap_INFO($parameter) ignore required type list
	    if { $type != {w} } {
		if { $list == 1 } { set list 2 }
		EvaP_set_value 1 $type  $list $evap_DEFAULT_VALUE($parameter) $parameter
	    }
	}; # ifend unevaluated
    }

    # Store program name for caller.

    EvaP_set_value 0 {w} 0 $argv0 {help}

    # Ensure all $required parameters have been specified on the command line.

    foreach p $evap_REQUIRED {
	puts stderr "Parameter $p is required but was omitted."
	incr evap_error 1
    }

    # Ensure any required files follow, or none do if that is the case.

    if { [regexp "$evap_pdt_reg_exp4" $evap_file_list] && $argc > 0 } {
	puts -nonewline stderr "$evap_Help_Hooks(P_HHENFL)"
	incr evap_error 1
    }
    if { [regexp "$evap_pdt_reg_exp6" $evap_file_list] && $argc == 0 } {
	puts -nonewline stderr "$evap_Help_Hooks(P_HHERFL)"
	incr evap_error 1
    }

    # Finish up.

    if { $evap_error > 0 } {	puts stderr "Type $argv0 -? for command line parameter information." }

    if { $evap_error > 0 && $evap_embed == 0 } { exit 1 }
    if { $evap_error == 0 } {
	return 1
    } else {
	return 0
    }

}; # end EvaP_fin




proc EvaP_PDT_error {msg} {

    #
    # Inform the application developer that they've screwed up!
    #

    global evap_error

    puts stderr "$msg"
    incr evap_error 1

}; # end EvaP_PDT_error




proc EvaP_set_value {evaluate type list v parameter} {
    
    #
    # Store a parameter's value; some parameter types require special type
    # conversion.  Store values the old way in scalar/list variables of the
    # form $opt_parameter, as well as the new way in the associative arrays
    # named $options() and $Options().  'list of' parameters are are returned
    # as a string 'joined' with the multi-dimensional array emulation character
    # \x1C.
    #
    # Evaluate items in grave accents (backticks), boolean and files if
    # `evaluate' is TRUE.
    #
    # Handle list syntax (item1, item2, ...) for 'list of' types.
    #
    # Lists are a little weird as they may already have default values from the
    # PDT declaration. The first time a list parameter is specified on the
    # command line we must first empty the list of its default values.  The
    # evap_INFO list flag thus can be in one of three states: 1 = the list has
    # possible default values from the PDT, 2 = first time for this command
    # line parameter so empty the list and THEN push the parameter's value, and
    # 3 = from now just keep pushing new command line values on the list.
    #

    global evap_QUAL
    global ${evap_QUAL}options ${evap_QUAL}Options evap_EVALUATE
    set parameter_old "${evap_QUAL}opt_$parameter"
    global $parameter_old evap_DOS evap_pdt_reg_exp2 evap_pdt_reg_exp3 env evap_d evap_D evap_s evap_S evap_w evap_W

    if { $list == 2 || \
      ( ! [info exists $parameter_old] && ! [info exists ${evap_QUAL}options($parameter)] &&
       ! [info exists ${evap_QUAL}Options($parameter)] ) } {
	set $parameter_old {}
	set ${evap_QUAL}options($parameter) {}
	set ${evap_QUAL}Options($parameter) {}
    }

    if { [regexp {^\(+(.*)\)+$} $v ignore v] } {; # check for list
	set values [EvaP_decompose_list $v]
    } else {; # not a list
        regexp {^['|"](.*)['|"]$} $v ignore v
	lappend values $v
    }

    foreach value $values {

	if { $evaluate } {

	    set evap_EVALUATE($parameter) {evaluated}
	    regexp {^(`*)([^`]*)(`*)$} $value ignore p1 p2 p3
	    if { $p1 == {`} && $p3 == {`} } {
		set f [open "| $p2" r]
		set value {}
		while { [gets $f line] != -1 } {
		    append value "$line\n"
		}
		set value [string trimright $value \n]
	    }

	    if { ! $evap_DOS && $type == "f" } {
		set path [split $value {/}]
		if { [lindex $path 0] == {~} || [lindex $path 0] == {$HOME} } {
		    set path [lreplace $path 0 0 $env(HOME)]
		    set value [join $path {/}]
		}
	    }

	    if { $type == "b" } {
		if { [regexp -nocase "$evap_pdt_reg_exp2" $value] } { set value 1 }
		if { [regexp -nocase "$evap_pdt_reg_exp3" $value] } { set value 0 }
	    }

	}; # ifend evaluate

	if { $list } {
	    lappend $parameter_old $value
	    if { [set ${evap_QUAL}options($parameter)] == {} && [set ${evap_QUAL}Options($parameter)] == {} } {
		set ${evap_QUAL}options($parameter) $value
		set ${evap_QUAL}Options($parameter) $value
	    } else {
		set ${evap_QUAL}options($parameter) "[set ${evap_QUAL}options($parameter)]\x1C$value"
		set ${evap_QUAL}Options($parameter) "[set ${evap_QUAL}Options($parameter)]\x1C$value"
	    }
	} else {
	    set $parameter_old $value
	    set ${evap_QUAL}options($parameter) $value
	    set ${evap_QUAL}Options($parameter) $value
	}
    }; # forend all values

}; # end EvaP_set_value




proc EvaP_decompose_list { v } {

    #
    # Parse a string in EvaP list notation (item1, item2, item-n) and return a Tcl list.
    # Adapted from Perl's shellword.pl code.
    #

    set v [string trimleft $v]

    while { $v != "" } {

	set field ""

	while { 1 == 1 } {
	    if { [regexp  {^"(([^"\\]|\\[\\"])*)"(.*)} $v ignore snippet v1 v] } {
		regsub -all {\\(.)} $snippet & snippet
	    } elseif { [regexp {^"} $v] } {
                puts stderr "Unmatched double quote: $v"
            } elseif { [regexp {^'(([^'\\]|\\[\\'])*)'(.*)} $v ignore snippet v1 v] } {
                regsub -all {\\(.)} $snippet & snippet
            } elseif { [regexp {^'} $v] } {
	        puts stderr "Unmatched single quote: $v"
            } elseif { [regexp {^\\(.)} $v snippet] } {
            } elseif { [regexp "^(\[^ \t\n\r\f\\'\"\]+)(.*)" $v ignore snippet v v1] } {
	        if { $snippet == {,} } { set snippet {} }
                set snippet [string trimright $snippet {,}]
            } else {
                set v [string trimleft $v]
		break
            }
            append field $snippet
	}; # whilend all snippets

        lappend values $field

    }; # whilend string is not empty

    return [expr { ([info exists values]) ? $values : "" } ]

}; # end EvaP_decompose_list




proc evap_pac {prompt I cmds0} {

    #
    # Process Application Commands
    #
    # An application command can be invoked by entering either its full spelling or the alias.
    #

    upvar $cmds0 cmds 
    global evap_embed evap_shell env argc argv0 argv evap_s evap_S

    set evap_embed 1;	# enable embedding
    set evap_shell [expr { ([info exists env(SHELL)] && ($env(SHELL) != "") ) ? $env(SHELL) : "/bin/sh" } ]
    set cmds(display_application_commands|disac) {evap_disac_proc cmds}
    set cmds(!) "evap_bang_proc"

    # First, create new associative command name arrays with full/alias names.

    foreach name [array names cmds] {
        if { [regexp {\|} $name] } {
            regexp {(.*)\|(.*)} $name junk l a
	    set long($l) $cmds($name)
	    set alias($a) $cmds($name)
        } else {
	    set long($name) $cmds($name)
	}
    }

    for {puts -nonewline stdout "$prompt"} {[gets $I line] != -1} {puts -nonewline stdout "$prompt"} {

        if { [regexp "^$evap_s*$" $line] } {
            continue
        }

	if { [regexp "^$evap_s*!(.+)" $line $junk new_line] } {
	    set line "! $new_line"
	}

        regexp "$evap_s*($evap_S+)$evap_s*(.*)" $line junk argv0 args
	if { [info exists long($argv0)] } {
	    set proc $long($argv0)
	} elseif { [info exists alias($argv0)] } {
	    set proc $alias($argv0)
	} else  {
            puts -nonewline stderr "Error - unknown command `$argv0'.  Type \"disac -do f\" for a\n"
	    puts -nonewline stderr "list of valid application commands.  You can then ...\n\n"
            puts -nonewline stderr "Type \"xyzzy -?\" for help on application command `xyzzy'.\n"
	    continue
        }
	
	if { $argv0 == "!" } {
	    set argv [join $args " "]
            set argc 1
	} else {
	    set argv [EvaP_decompose_list $args]
            set argc [llength $argv]
	}
        eval "$proc";		# call the evap/user procedure

    }
    puts -nonewline stdout "\n"

}; # end evap_pac

 


proc evap_bang_proc {} {
    
    #
    # Issue one or more commands to the user's shell.  If the SHELL environment
    # variable is not defined or is empty, then /bin/sh is used.
    #

    global argc argv0 argv evap_shell evap_Help_Hooks

    if { $argv == "" } { return }
    set cmd $argv

    set Q "bang_pkg_"; evap_setup_for_evap "$Q" "bang"
    global ${Q}PDT ${Q}MM ${Q}Options
    set evap_Help_Hooks(P_HHUOFL) " Command(s)\n"
    set evap_Help_Hooks(P_HHBOFL) "\nA list of shell Commands.\n\n"
    if { [evap [set ${Q}PDT] [set ${Q}MM] $Q] != 1 } { return }

    if { [catch {puts stdout [exec $evap_shell "-c" "$cmd"]} msg] != 0 } {
        puts stdout $msg
    }

}; # end evap_bang_proc

 


proc evap_disac_proc { commands0 } {
    
    #
    # Display the list of legal application commands.
    #

    global argc argv0 argv

    upvar $commands0 commands

    set Q "disac_pkg_"; evap_setup_for_evap "$Q" "disac"
    global ${Q}PDT ${Q}MM ${Q}Options
    if { [evap [set ${Q}PDT] [set ${Q}MM] $Q] != 1 } { return }

    foreach name [array names commands] {
        if { [regexp {\|} $name] } {
            regexp {(.*)\|(.*)} $name junk l a
        } else {
	    set l $name
            set a ""
	}
        lappend brief $l
        lappend full [expr { ($a != "") ? "$l, $a" : "$l" } ]
    }

    if { [set ${Q}Options(output)] == "stdout" } {
        set H [set ${Q}Options(output)]
    } else {
        set H [open [set ${Q}Options(output)] "w"]
    }
    if { [set ${Q}Options(display_option)] == "full" } {
        puts -nonewline $H "\nCommands and aliases for this application:\n\n"
	puts -nonewline $H "  [join [lsort $full] "\n  "]\n"
    } else {
        puts -nonewline $H "[join [lsort $brief] "\n"]\n"
    }
    if { $H != "stdout" } {
	close $H
    }

}; # end evap_disac_proc




proc evap_setup_for_evap {Q command} { 

    #
    # Initialize evap_pac's builtin commands' PDT/MM variables.
    #

    global ${Q}PDT ${Q}MM evap_message_modules

    set IN [open "|ar p $evap_message_modules ${command}_pdt" "r"]
    set ${Q}PDT [split [read $IN] \n];                      # initialize Parameter Description Table
    close $IN
    set IN [open "|ar p $evap_message_modules ${command}.mm" "r"]
    set ${Q}MM [split [string trimright [read $IN] \n] \n]; # initialize Message Module
    close $IN

}; # end setup_for_evap
