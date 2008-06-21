#!/bin/sh
#-*-tcl-*-
# the next line restarts using wish \
exec wish "$0" -- ${1+"$@"}

###############################################################################
#
# TkDiff -- A graphical front-end to diff for Unix and Windows.
# Copyright (C) 1994-1998 by John M. Klassa.
# Copyright (C) 1999-2001 by AccuRev Inc.
# Copyright (C) 2002-2005 by John M. Klassa.
#
# TkDiff Home Page: http://tkdiff.sourceforge.net
#
# Usage:  see "tkdiff -h" or "tkdiff --help"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.        See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
###############################################################################

package require Tk 8.0

# Change to t for trace info on stderr
set g(debug) f

# get this out of the way -- we want to draw the whole user interface
# behind the scenes, then pop up in all of its well-laid-out glory
set screenWidth [winfo vrootwidth .]
set screenHeight [winfo vrootheight .]
wm withdraw .

# set a couple o' globals that we might need sooner than later
set g(name) "TkDiff"
set g(version) "4.1.4"
set g(started) 0

# FIXME - move to preferences
option add "*TearOff" false 100
option add "*BorderWidth" 1 100
option add "*ToolTip.background" LightGoldenrod1
option add "*ToolTip.foreground" black

# determine the windowing platform, since there are different ways to
# do this for different versions of tcl
if {[catch {tk windowingsystem} g(windowingSystem)]} {
    if {"$::tcl_platform(platform)" == "windows"} {
        set g(windowingSystem) "win32"
    } elseif {"$::tcl_platform(platform)" == "unix"} {
        set g(windowingSystem) "x11"
    } elseif {"$::tcl_platform(platform)" == "macintosh"} {
        set g(windowingSystem) "classic"
    } else {
        # this should never happen, but just to be sure...
        set g(windowingSystem) "x11"
    }
}

# determine the name of the temporary directory and the name of
# the rc file, both of which are dependent on the platform.
# This is overridden by the preference in .tkdiffrc except for the very first
# time you run
switch -- $::tcl_platform(platform) {
windows {
        if {[info exists env(TEMP)]} {
            set opts(tmpdir) [file nativename $env(TEMP)]
        } else {
            set opts(tmpdir) C:/temp
        }
        set basercfile "_tkdiff.rc"
        # Native look for toolbar
        set opts(fancyButtons) 1
        set opts(relief) flat
    }
default {
        if {[info exists env(TMPDIR)]} {
            set opts(tmpdir) $env(TMPDIR)
        } else {
            set opts(tmpdir) /tmp
        }
        set basercfile ".tkdiffrc"
        # Native look for toolbar
        set opts(fancyButtons) 0
        set opts(relief) raised
    }
}

# compute preferences file location. Note that TKDIFFRC can hold either
# a directory or a file, though we document it as being a file name
if {[info exists env(TKDIFFRC)]} {
    set rcfile $env(TKDIFFRC)
    if {[file isdirectory $rcfile]} {
        set rcfile [file join $rcfile $basercfile]
    }
} elseif {[info exists env(HOME)]} {
    set rcfile [file join $env(HOME) $basercfile]
} else {
    set rcfile [file join "/" $basercfile]
}

# Try to find a pleasing native look for each platform.
# Fonts.
set sysfont [font actual system]
#debug-info "system font: $sysfont"

# See what the native menu font is
. configure -menu .native
menu .native
set menufont [lindex [.native configure -font] 3]
destroy .native

# Find out what the tk default is
label .testlbl -text "LABEL"
set labelfont [lindex [.testlbl configure -font] 3]
destroy .testlbl

text .testtext
set textfont [lindex [.testtext configure -font] 3]
destroy .testtext

entry .testent
set w(selcolor) [lindex [.testent configure -selectbackground] 4]
set entryfont [lindex [.testent configure -font] 3]
destroy .testent
# the above results in a nearly undistinguishable darker gray for the
# selected color (rh8 with tk 8.3.3-74) "#c3c3c3"
set w(selcolor) "#b03060"

#debug-info "menufont $menufont"
#debug-info "labelfont $labelfont"
#debug-info "textfont $textfont"
#debug-info "entryfont $entryfont"

set fs [lindex $textfont 1]
if {$fs == ""} {
  # This happens on Windows in tk8.5
  # You get {TkDefaultFont} instead of {fixed 12} or whatever
  # Then when you add "bold" to it you have a bad spec
  set fa [font actual $textfont]
  puts " actual font: $fa"
  set fm [lindex $fa 1]
  set fs [lindex $fa 3]
  set textfont [list $fm $fs]
}
set font [list $textfont]
set bold [list [concat $textfont bold]]
#debug-info "font: $font"
#debug-info "bold: $bold\n"
option add *Label.font $labelfont userDefault
option add *Button.font $labelfont userDefault
option add *Menu.font $menufont userDefault
option add *Entry.font $entryfont userDefault

# This makes tk_messageBox use our font.  The default tends to be terrible
# no matter what platform
option add *Dialog.msg.font $labelfont userDefault

# Initialize arrays
array set g {
    ancfileset      0
    conflictset     0
    ancfile         ""
    changefile      "tkdiff-change-bars.out"
    destroy         ""
    ignore_event,1  0
    ignore_event,2  0
    ignore_hevent,1 0
    ignore_hevent,2 0
    initOK          0
    mapborder       0
    mapheight       0
    mergefile       ""
    returnValue     0
    showmerge       0
    started         0
    mergefileset    0
    tempfiles       ""
    thumbMinHeight  10
    thumbHeight     10
    thumbDeltaY     0
}

array set finfo {
    f,1          ""
    f,2          ""
    pth,1        ""
    pth,2        ""
    revs,1       ""
    revs,2       ""
    lbl,1        ""
    lbl,2        ""
    userlbl,1    ""
    userlbl,2    ""
    title        {}
    tmp,1        0
    tmp,2        0
}
set uniq 0

# These options may be changed at runtime
array set opts {
    autocenter        1
    autoselect        0
    colorcbs          0
    customCode        {}
    diffcmd           "diff"
    ignoreblanksopt   "-b"
    ignoreblanks      0
    editor            ""
    geometry          "80x30"
    showcbs           1
    showln            1
    showmap           1
    showlineview      0
    showinline1       0
    showinline2       1
    syncscroll        1
    toolbarIcons      1
    tagcbs            0
    tagln             0
    tagtext           1
    tabstops          8
}

# reporting options
array set report {
    doSideLeft                0
    doLineNumbersLeft         1
    doChangeMarkersLeft       1
    doTextLeft                1
    doSideRight               1
    doLineNumbersRight        1
    doChangeMarkersRight      1
    doTextRight               1
    filename                  "tkdiff.out"
}

if {[string first "color" [winfo visual .]] >= 0} {
    # We have color
    # (but, let's not go crazy...)

    set colordel Tomato
    set colorins PaleGreen
    set colorchg DodgerBlue

    array set opts [subst {
        textopt    "-background white -foreground black -font $font"
        currtag    "-background Khaki"
        difftag    "-background gray"
        deltag     "-background $colordel -font $bold"
        instag     "-background $colorins -font $bold"
        chgtag     "-background LightSteelBlue"
        overlaptag "-background yellow"
        bytetag    "-background blue -foreground white"
        inlinetag  "-background $colorchg -font $bold"
        -          "-background $colordel -foreground $colorins"
        +          "-background $colorins -foreground $colordel"
        !          "-background $colorchg -foreground $colorchg"
        ?          "-background yellow -foreground yellow"
        mapins     "$colorins"
        mapdel     "$colordel"
        mapchg     "$colorchg"
    }]

} else {
    # Assume only black and white
    set bg "black"
    array set opts [subst {
        textopt    "-background white -foreground black -font $font"
        currtag    "-background black -foreground white"
        difftag    "-background white -foreground black -font $bold"
        deltag     "-background black -foreground white"
        instag     "-background black -foreground white"
        chgtag     "-background black -foreground white"
        overlaptag "-background black -foreground white"
        bytetag    "-underline 1"
        inlinetag  "-underline 1"
        -          "-background black -foreground white"
        +          "-background black -foreground white"
        !          "-background black -foreground white"
        ?          "-background black -foreground white"
        mapins     "black"
        mapdel     "black"
        mapchg     "black"
    }]
}

# make sure wrapping is turned off. This might piss off a few people,
# but it would screw up the display to have things wrap
set opts(textopt) "$opts(textopt) -wrap none"

# This proc is used in the rc file
proc define {name value} {
    global opts
    set opts($name) $value
}

# Source the rc file, which may override some of the defaults
# Any errors will be reported. Before doing so, we need to define the
# "define" proc, which lets the rc file have a slightly more human-friendly
# interface. Old-style .rc files should still load just fine for now, though
# it ought to be noted new .rc files won't be able to be processed by older
# versions of TkDiff. That shouldn't be a problem.
if {[file exists $rcfile]} {
    if {[catch {source $rcfile} error]} {
        set startupError [join [list "There was an error in processing your \
          startup file." "\n$g(name) will still run, but some of your \
          preferences" "\nmay not be in effect." "\n\nFile: $rcfile" \
          "\nError: $error"] " "]
    }
}

# a hack to handle older preferences files...
# if the user has a diffopt defined in their rc file, we'll magically
# convert that to diffcmd...
if {[info exists opts(diffopt)]} {
    set opts(diffcmd) "diff $opts(diffopt)"
}

# Work-around for bad font approximations,
# as suggested by Don Libes (libes@nist.gov).
catch {tk scaling [expr {100.0 / 72}]}

###############################################################################
#
# HERE BEGIN THE PROCS
###############################################################################

###############################################################################
# Exit with proper code
###############################################################################
proc do-exit {{returncode {}}} {
    debug-info "do-exit ($returncode)"
    global g

    # we don't particularly care if del-tmp fails.
    catch {del-tmp}
    if {$returncode == ""} {
        set returncode $g(returnValue)
    }
    # exit with an appropriate return value
    exit $returncode
}

###############################################################################
# Modal error dialog.
###############################################################################
proc do-error {msg} {
    global g

    debug-info "do-error ($msg)"
    tk_messageBox -message "$msg" -title "$g(name): Error" -icon error -type ok
}

###############################################################################
# Throw up a modal error dialog or print a message to stderr.  For
# Unix we print to stderr and exit if the main window hasn't been
# created, otherwise put up a dialog and throw an exception.
###############################################################################
proc fatal-error {msg} {
    debug-info "fatal-error ($msg)"
    global g tcl_platform

    if {$g(started)} {
        tk_messageBox -title "Error" -icon error -type ok -message $msg
        do-exit 2
    } else {
        puts stderr $msg
        del-tmp
        do-exit 2
    }
}

###############################################################################
# Return the name of a temporary file
###############################################################################
proc tmpfile {n} {
    debug-info "tmpfile ($n)"
    global g opts
    global uniq
    set uniq [expr ($uniq + 1) ]
    set tmpdir [file nativename $opts(tmpdir)]
    set tmpfile [file join $tmpdir "[pid]-$n-$uniq"]
    set access [list RDWR CREAT EXCL TRUNC]
    set perm 0600
    if {[catch {open $tmpfile $access $perm} fid ]} {
        # something went wrong
        error "Failed creating temporary file: $fid"
    }
    close $fid
    lappend g(tempfiles) $tmpfile
    return $tmpfile
}

###############################################################################
# Execute a command.
# Returns "$stdout $stderr $exitcode" if exit code != 0
###############################################################################
proc run-command {cmd} {
    debug-info "run-command ($cmd)"
    global opts errorCode

    set stderr ""
    set exitcode 0
    set errfile [tmpfile "r"]

    set failed [catch "$cmd \"2>$errfile\"" stdout]
    # Read stderr output
    catch {
        set hndl [open "$errfile" r]
        set stderr [read $hndl]
        close $hndl
    }
    if {$failed} {
        switch -- [lindex $errorCode 0] {
        "CHILDSTATUS" {
                set exitcode [lindex $errorCode 2]
            }
        "POSIX" {
                if {$stderr == ""} {
                    set stderr $stdout
                }
                set exitcode -1
            }
        default {
                set exitcode -1
            }
        }
    }

    catch {file delete $errfile}
    return [list "$stdout" "$stderr" "$exitcode"]
}

###############################################################################
# Execute a command.  Die if unsuccessful.
###############################################################################
proc die-unless {cmd file} {
    #debug-info "die-unless ($cmd $file)"
    global opts errorCode

    set file [string trim $file "\""]
    set result [run-command "$cmd \">$file\""]
    set stdout [lindex $result 0]
    set stderr [lindex $result 1]
    set exitcode [lindex $result 2]

    if {$exitcode != 0} {
        fatal-error "$stderr\n$stdout"
    }
}

###############################################################################
# Filter PVCS output files that have CR-CR-LF end-of-lines
###############################################################################
proc filterCRCRLF {file} {
    debug-info "filterCRCLF ($file)"
    set outfile [tmpfile 9]
    set inp [open $file r]
    set out [open $outfile w]
    fconfigure $inp -translation binary
    fconfigure $out -translation binary
    set CR [format %c 13]
    while {![eof $inp]} {
        set line [gets $inp]
        if {[string length $line] && ![eof $inp]} {
            regsub -all "$CR$CR" $line $CR line
            puts $out $line
        }
    }
    close $inp
    close $out
    file rename -force $outfile $file
}

###############################################################################
# Return the smallest of two values
###############################################################################
proc min {a b} {
    return [expr {$a < $b ? $a : $b}]
}

###############################################################################
# Return the largest of two values
###############################################################################
proc max {a b} {
    return [expr {$a > $b ? $a : $b}]
}

###############################################################################
# Toggle change bars
###############################################################################
proc do-show-changebars {{show {}}} {
    debug-info "do-show-changebars ($show)"
    global opts
    global w

    if {$show != {}} {
        set opts(showcbs) $show
    }

    if {$opts(showcbs)} {
        grid $w(LeftCB) -row 0 -column 2 -sticky ns
        grid $w(RightCB) -row 0 -column 1 -sticky ns
    } else {
        grid forget $w(LeftCB)
        grid forget $w(RightCB)
    }
}

###############################################################################
# Toggle ignore white spaces
###############################################################################
proc do-show-ignoreblanks {{showIgn {}}} {
    global opts
    global finfo

    if {$showIgn != {}} {
        set opts(ignoreblanks) $showIgn
    }
    if {$finfo(pth,1) != {} && $finfo(pth,2) != {}} {
        recompute-diff
    }
}

###############################################################################
# Toggle line numbers.
###############################################################################
proc do-show-linenumbers {{showLn {}}} {
    global opts
    global w

    if {$showLn != {}} {
        set opts(showln) $showLn
    }

    if {$opts(showln)} {
        grid $w(LeftInfo) -row 0 -column 1 -sticky nsew
        grid $w(RightInfo) -row 0 -column 0 -sticky nsew
    } else {
        grid forget $w(LeftInfo)
        grid forget $w(RightInfo)
    }
}

###############################################################################
# Show line numbers in info windows
###############################################################################
proc draw-line-numbers {} {
    global g
    global w

    $w(LeftInfo) configure -state normal
    $w(RightInfo) configure -state normal
    $w(LeftCB) configure -state normal
    $w(RightCB) configure -state normal

    set lines(Left) [lindex [split [$w(LeftText) index end-1lines] .] 0]
    set lines(Right) [lindex [split [$w(RightText) index end-1lines] .] 0]

    # Smallest line count
    set minlines [min $lines(Left) $lines(Right)]

    # cache all the blank lines for the info and cb windows, and do
    # one big insert after we're done. This seems to be much quicker
    # than inserting them in the widgets one line at a time.
    set linestuff {}
    set cbstuff {}
    for {set i 1} {$i < $minlines} {incr i} {
        append linestuff "$i\n"
        append cbstuff " \n" ;# for now, just put in place holders...
    }

    $w(LeftInfo) insert end $linestuff
    $w(RightInfo) insert end $linestuff
    $w(LeftCB) insert end $cbstuff
    $w(RightCB) insert end $cbstuff

    # Insert remaining line numbers. We'll cache the stuff to be
    # inserted so we can do just one call in to the widget. This
    # should be much faster, relatively speaking, then inserting
    # data one line at a time.
    foreach mod {Left Right} {
        set linestuff {}
        set cbstuff {}
        for {set i $minlines} {$i < $lines($mod)} {incr i} {
            append linestuff "$i\n"
            append cbstuff " \n" ;# for now, just put in place holders...
        }
        $w(${mod}Info) insert end $linestuff
        $w(${mod}CB) insert end $cbstuff
    }

    $w(LeftCB) configure -state disabled
    $w(RightCB) configure -state disabled

    $w(LeftInfo) configure -state disabled
    $w(RightInfo) configure -state disabled
}

###############################################################################
# Pop up a window for file merge.
###############################################################################
proc popup-merge {{writeproc merge-write-file}} {
    debug-info "popup-merge ($writeproc)"
    global g
    global w

    if {$g(mergefileset)} {
        $writeproc
        return
    }

    set types {
        {{Text Files}         {.txt}}
        {{All Files}         {*}}
    }

    set path [tk_getSaveFile -defaultextension "" -filetypes $types \
      -initialfile [file nativename $g(mergefile)]]

    if {[string length $path] > 0} {
        set g(mergefile) $path
        $writeproc
    }
}

###############################################################################
# Split a file containing CVS conflict markers into two temporary files
#    name        Name of file containing conflict markers
# Returns the names of the two temporary files and the names of the
# files that were merged
###############################################################################
proc split-conflictfile {name} {
    debug-info "conflicts ($name)"
    global g opts

    set first ${name}.1
    set second ${name}.2

    set temp1 [tmpfile 1]
    set temp2 [tmpfile 2]

    if {[catch {set input [open $name r]}]} {
        fatal-error "Couldn't open file '$name'"
    }
    set first [open $temp1 w]
    set second [open $temp2 w]

    set firstname ""
    set secondname ""
    set output 3

    set firstMatch ""
    set secondMatch ""
    set thirdMatch ""

    while {[gets $input line] >= 0} {
        if {$firstMatch == ""} {
            if {[regexp {^<<<<<<<* +(.*)} $line]} {
                set firstMatch {^<<<<<<<* +(.*)}
                set secondMatch {^=======*}
                set thirdMatch {^>>>>>>>* +(.*)}
            } elseif {[regexp {^>>>>>>>* +(.*)} $line]} {
                set firstMatch {^>>>>>>>* +(.*)}
                set secondMatch {^<<<<<<<* +(.*)}
                set thirdMatch {^=======*}
            }
        }
        if {$firstMatch != ""} {
            if {[regexp $firstMatch $line]} {
                set output 2
                if {$secondname == ""} {
                    regexp $firstMatch $line all secondname
                }
            } elseif {[regexp $secondMatch $line]} {
                set output 1
                if {$firstname == ""} {
                    regexp $secondMatch $line all firstname
                }
            } elseif {[regexp $thirdMatch $line]} {
                set output 3
                if {$firstname == ""} {
                    regexp $thirdMatch $line all firstname
                }
            } else {
                if {$output & 1} {
                    puts $first $line
                }
                if {$output & 2} {
                    puts $second $line
                }
            }
        } else {
            puts $first $line
            puts $second $line
        }
    }
    close $input
    close $first
    close $second

    if {$firstname == ""} {
        set firstname "old"
    }
    if {$secondname == ""} {
        set secondname "new"
    }

    return "{$temp1} {$temp2} {$firstname} {$secondname}"
}

###############################################################################
# Get a revision of a file
#   f            file name
#   index   index in finfo array
#   r            revision, "" for head revision
###############################################################################
proc get-file-rev {f index {r ""}} {
    debug-info "get-file-rev ($f $index \"$r\")"
    global finfo
    global opts
    global tcl_platform

    if {"$r" == ""} {
        set rev "HEAD"
        set acrev "HEAD"
        set acopt ""
        set cvsopt ""
        set svnopt ""
        set rcsopt ""
        set sccsopt ""
        set bkopt ""
        set pvcsopt ""
        set p4file "$f"
    } else {
        set rev "r$r"
        set acrev "\"$r\""
        set acopt "-v \"$r\""
        set cvsopt "-r $r"
        set svnopt "-r $r"
        set rcsopt "$r"
        set sccsopt "-r$r"
        set bkopt "-r$r"
        set pvcsopt "-r$r"
        set p4file "$f#$r"
    }

    set finfo(pth,$index) [tmpfile $index]
    set finfo(tmp,$index) 1

    # NB: it would probably be a Good Thing to move the definition
    # of the various command to exec, to the preferences dialog.

    regsub -all {\$} $f {\$} f
    set dirname [file dirname $f]
    set tailname [file tail $f]

    debug-info " $f"
    # For CVS, if it isn't checked out there is neither a CVS nor RCS
    # directory.  It will however have a ,v suffix just like rcs.
    # There is not necessarily a RCS directory for RCS, either. The file
    # always has a ,v suffix.

    if {[file isdirectory [file join $dirname CVS]]} {
        set cmd "cvs"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        set finfo(lbl,$index) "$f (CVS $rev)"
        debug-info "  Setting lbl $finfo(lbl,$index)"
        die-unless "exec $cmd update -p $cvsopt \"$f\"" "\"$finfo(pth,$index)\""
    } elseif {[file isdirectory [file join $dirname .svn]]} {
        set cmd "svn"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        if {"$r" == "" || "$rev" == "rBASE"} {
            set finfo(lbl,$index) "$f (SVN BASE)"
            debug-info "  Setting lbl $finfo(lbl,$index)"
            die-unless "exec cat \"$dirname/.svn/text-base/$tailname.svn-base\"" \
              $finfo(pth,$index)
        } else {
            set finfo(lbl,$index) "$f (SVN $rev)"
            debug-info "  Setting lbl $finfo(lbl,$index)"
            die-unless "exec $cmd cat $svnopt \"$f\"" $finfo(pth,$index)
        }
    } elseif {[regexp {://} $f]} {
        # Subversion command can have the form
        # svn diff OLD-URL[@OLDREV] NEW-URL[@NEWREV]
        if {![regsub {^.*@} $f {} rev]} {
            set rev "HEAD"
        }
        regsub {@\d+$} $f {} path
        set finfo(lbl,$index) "$f"
        set cmd "svn"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        if {"$rev" == ""} {
            set command "$cmd cat $path"
        } else {
            set command "$cmd cat -r$rev $path"
        }
        die-unless "exec $command" $finfo(pth,$index)
    } elseif {[file isdirectory [file join $dirname SCCS]]} {
        if {[sccs-is-bk]} {
            set cmd "bk"
            set opt $bkopt
            set finfo(lbl,$index) "$f (bitkeeper $rev)"
            debug-info "  Setting lbl $finfo(lbl,$index)"
        } else {
            set finfo(lbl,$index) "$f (SCCS $rev)"
            debug-info "  Setting lbl $finfo(lbl,$index)"
            set opt $sccsopt
            set cmd "sccs"
        }
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        die-unless "exec $cmd get -p $opt \"$f\"" "\"$finfo(pth,$index)\""
    } elseif {[file isdirectory [file join $dirname RCS]]} {
        set cmd "co"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        set finfo(lbl,$index) "$f (RCS $rev)"
        debug-info "  Setting lbl $finfo(lbl,$index)"
        die-unless "exec $cmd -p$rcsopt \"$f\"" "\"$finfo(pth,$index)\""
    } elseif {[file exists [file join $dirname $tailname,v]]} {
        set cmd "co"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        set finfo(lbl,$index) "$f (RCS $rev)"
        debug-info "  Setting lbl $finfo(lbl,$index)"
        die-unless "exec $cmd -p$rcsopt \"$f\"" \""$finfo(pth,$index)\""
    } elseif {[file exists [file join $dirname vcs.cfg]]} {
        set cmd "get"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        set finfo(lbl,$index) "$f (PVCS $rev)"
        debug-info "  Setting lbl $finfo(lbl,$index)"
        die-unless "exec $cmd -p $pvcsopt \"$f\"" "\"$finfo(pth,$index)\""
        filterCRCRLF $finfo(pth,$index)
    } elseif {[info exists ::env(P4CLIENT)] || [info exists ::env(P4CONFIG)]} {
        set cmd "p4"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        set finfo(lbl,$index) "$f (Perforce $rev)"
        debug-info "  Setting lbl $finfo(lbl,$index)"
        die-unless "exec $cmd print -q \"$p4file\"" "\"$finfo(pth,$index)\""
    } elseif {[info exists ::env(ACCUREV_BIN)]} {
        set cmd "accurev"
        if {$::tcl_platform(platform) == "windows"} {
            append cmd ".exe"
        }
        set finfo(lbl,$index) "$f ($acrev)"
        debug-info "  Setting lbl $finfo(lbl,$index)"
        die-unless "exec $cmd cat $acopt \"$f\"" "\"$finfo(pth,$index)\""
    } elseif {[info exists ::env(CLEARCASE_ROOT)]} {
        set cmd "cleartool"
        set finfo(lbl,$index) "$f (ClearCase $rev)"
        debug-info "  Setting lbl $finfo(lbl,$index)"
        catch {exec $cmd ls -s $f} ctls
        # get the path name to file minus the revision info
        # (either CHECKEDOUT or a number)
        if {![regexp {(\S+)/([^/]+)$} $ctls dummy path checkedout]} {
            puts "Couldn't parse ct ls output '$ctls'"
            exit
        }
        catch {exec $cmd lshistory -last 50 $f} ctlshistory
        set lines [split $ctlshistory "\n"]
        set predecessor ""
        # find the previous version
        if {$checkedout == "CHECKEDOUT" || $checkedout == 0} {
            if {$checkedout == 0} {
                set path [file dirname $path]
            }
            set pattern "create version \"($path/\[^/\]+)\""
        } else {
            incr checkedout -1
            set pattern "create version \"($path/$checkedout)\""
        }
        # search the history of the file for the latest version on our branch
        foreach l $lines {
            if {[regexp $pattern $l dummy predecessor]} {
                break
            }
        }
        if {$predecessor != ""} {
            set finfo(pth,$index) $predecessor
            debug-info "  Setting lbl from predecessor $finfo(lbl,$index)"
        } else {
            puts "Couldn't deal with $f, exiting..."
            exit
        }
    } else {
        fatal-error "File '$f' is not part of a revision control system"
    }
    # Header above each file - if user has specified -L, override
    #debug-info "  $finfo(lbl,$index)"
    if {$finfo(userlbl,$index) != {}} {
        set finfo(lbl,$index) $finfo(userlbl,$index)
        debug-info "  User label: $finfo(lbl,$index)"
    }
}

proc sccs-is-bk {} {
    set cmd [auto_execok "bk"]
    set result 0
    if {[string length $cmd] > 0} {
        if {![catch {exec bk root} error]} {
            set result 1
        }
    }
    return $result
}

###############################################################################
# Setup ordinary file
#   f            file name
#   index   index in finfo array
###############################################################################
proc get-file {f index} {
    debug-info "get-file ($f $index)"
    global finfo

    #set finfo(f,$index) $f
    if {[file exists $f] != 1} {
        fatal-error "File '$f' does not exist"
        return 1
    }
    if {[file isdirectory $f]} {
        fatal-error "'$f' is a directory"
        return 1
    }

    # Header above each file - use filename unless
    # user has specified one with -L
    set finfo(lbl,$index) "$f"
    debug-info "  Setting lbl  $finfo(lbl,$index)"
    if {$finfo(userlbl,$index) != {}} {
        set finfo(lbl,$index) $finfo(userlbl,$index)
        debug-info "  User label: $finfo(lbl,$index)"
    }
    set finfo(pth,$index) "$f"
    set finfo(tmp,$index) 0
    return 0
}

###############################################################################
# Read the commandline
###############################################################################
proc commandline {} {
    debug-info "commandline"
    global argv
    global argc
    debug-info "  argv: $argv"
    global finfo
    global opts
    global g

    set g(initOK) 0
    set argindex 0
    set revs 0
    set pths 0
    set lbls 0

    # Loop through argv, storing revision args in rev and file args in
    # finfo. revs and pths are counters.
    while {$argindex < $argc} {
        set arg [lindex $argv $argindex]
        switch -regexp -- $arg {
        "^-h" -
        "^--help" {
                do-usage cline
                exit 0
            }
        "^-a$" {
                incr argindex
                set g(ancfile) [lindex $argv $argindex]
            }
        "^-a.*" {
                set g(ancfile) [string range $arg 2 end]
            }
        "^-v$" -
        "^-r$" {
                incr argindex
                incr revs
                set finfo(revs,$revs) [lindex $argv $argindex]
            }
        "^-v.*" -
        "^-r.*" {
                incr revs
                set finfo(revs,$revs) [string range $arg 2 end]
            }
        "^-L$" {
                incr argindex
                incr lbls
                set finfo(userlbl,$lbls) [lindex $argv $argindex]
            }
        "^-L.*" {
                incr lbls
                set finfo(userlbl,$lbls) [string range $arg 2 end]
            }
        "^-conflict$" {
                set g(conflictset) 1
            }
        "^-o$" {
                incr argindex
                set g(mergefile) [lindex $argv $argindex]
            }
        "^-o.*" {
                set g(mergefile) [string range $arg 2 end]
            }
        "^-u$"  {
                # Ignore flag from "svn diff --diff-cmd=tkdiff"
            }
        "^-psn" {
                # Ignore the Carbon Process Serial Number
                set argv [lreplace $argv $argindex $argindex]
                incr argc -1
                incr argindex
            }
        "^-" {
                append opts(diffcmd) " $arg "
            }
        default {
                incr pths
                set finfo(pth,$pths) $arg
                set finfo(f,$pths) $arg
            }
        }
        incr argindex
    }

    # Add our counters to the global array
    # Now check how many revision and file args we have.
    debug-info "  $pths files, $revs revisions"
    # Maybe adjustment is needed
    if {$revs == 1 && $pths == 0} {
       # tkdiff -r FILE; same as tkdiff FILE
       set finfo(pths,1) $finfo(revs,1)
       set finfo(f,1) $finfo(revs,1)
       incr pths 1
       incr revs -1
       unset finfo(revs,1)
    } elseif {$revs == 2 && $pths == 0} {
       # tkdiff -rREV -r FILE; same as tkdiff -rREV FILE
       set finfo(pths,1) $finfo(revs,2)
       set finfo(f,1) $finfo(revs,2)
       incr pths 1
       incr revs -1
       unset finfo(revs,2)
    }
    # What have we got now?
    debug-info "  $pths files, $revs revisions"
    if {$revs == 0 && $pths == 0} {
        # Return "empty" flag, and we'll do a pop-up
        return 1
    } elseif {$revs > 1 && $pths != 1} {
        puts stderr "Error: you specified $pths file(s) and $revs revision(s)"
        do-usage cline
        exit 1
    }

    if {$g(mergefile) != ""} {
      set g(mergefileset) 1
    }
    return 0
}

###############################################################################
# Process the arguments, whether from the command line or from the dialog
###############################################################################
proc assemble-args {} {
    debug-info "assemble-args"
    global finfo
    global opts
    global g

    if {$g(ancfile) != ""} {
        set g(ancfileset) 1
    }
    debug-info " conflict: $g(conflictset)"
    debug-info " ancestor: $g(ancfileset) $g(ancfile)"
    debug-info " mergefile set: $g(mergefileset) $g(mergefile)"
    debug-info " diff command: $opts(diffcmd) "

    # Count up how many files and revs we got from the GUI or commandline
    set pths 0
    foreach p [array names finfo f,*] {
        if {$finfo($p) != ""} {
            incr pths
        }
    }
    set revs 0
    foreach r [array names finfo revs,*] {
        if {$finfo($r) != ""} {
            incr revs
        }
    }

    debug-info " $pths files, $revs revisions"
    if {$revs == 0 && $pths == 0} {
        return
    }
    if {$g(conflictset)} {
        if {$revs == 0 && $pths == 1} {
            ############################################################
            # tkdiff -conflict FILE
            ############################################################
            set files [split-conflictfile "$finfo(f,1)"]
            if {[get-file [lindex "$files" 0] 1]} {return}
            if {[get-file [lindex "$files" 1] 2]} {return}
            # A conflict file may come from merge, cvs, or vmrg.  The
            # names of the files/revisions depend on how it was made and
            # are taken from the <<<<<<< and >>>>>>> lines inside it.
            set finfo(lbl,1) [lindex "$files" 2]
            set finfo(lbl,2) [lindex "$files" 3]
        } else {
            fatal-error "Usage: tkdiff -conflict FILE"
        }
    } else {
        if {$revs == 2 && $pths == 1} {
            ############################################################
            #  tkdiff -rREV1 -rREV2 FILE
            ############################################################
            set f $finfo(f,1)
            get-file-rev "$f" 1 "$finfo(revs,1)"
            get-file-rev "$f" 2 "$finfo(revs,2)"

        } elseif {$revs == 1 && $pths == 1} {
            ############################################################
            #  tkdiff -rREV FILE
            ############################################################
            set f $finfo(f,1)
            get-file-rev "$f" 1 "$finfo(revs,1)"
            if {[get-file "$f" 2]} {return}

        } elseif {$revs == 0 && $pths == 2} {
            ############################################################
            #  tkdiff FILE1 FILE2
            ############################################################
            set f1 $finfo(f,1)
            set f2 $finfo(f,2)
            if {[file isdirectory $f1] && [file isdirectory $f2]} {
                fatal-error "Cannot diff two directories"
            }

            if {[file isdirectory $f1]} {
                set f1 [file join $f1 [file tail $f2]]
            } elseif {[file isdirectory $f2]} {
                set f2 [file join $f2 [file tail $f1]]
            }

            # Maybe they're Subversion URL paths, not local files
            if {[regexp {://} $f1]} {
                get-file-rev "$f1" 1
            } else {
                if {[get-file "$f1" 1]} {return}
            }
            if {[regexp {://} $f2]} {
                get-file-rev "$f2" 2
            } else {
                if {[get-file "$f2" 2]} {return}
            }


        } elseif {$revs == 0 && $pths == 1} {
            ############################################################
            #  tkdiff FILE
            ############################################################
            set f $finfo(f,1)
            get-file-rev "$f" 1
            if {[get-file "$f" 2]} {return}

        } else {
            do-error "Error: you specified $pths file(s) and $revs revision(s)"
            do-usage gui
            tkwait window .usage
            return 1
        }
    }

    set finfo(title) "[file tail $finfo(lbl,1)] vs. [file tail $finfo(lbl,2)]"
    debug-info "  Setting title $finfo(title)"
    set rootname [file rootname $finfo(pth,1)]
    #    set path          [file dirname          $finfo(pth,1)]
    set path [pwd]
    set suffix [file extension $finfo(pth,1)]
    if {! $g(mergefileset)} {
        set g(mergefile) [file join $path "${rootname}-merge$suffix"]
    }
    set g(initOK) 1
    foreach inf [lsort [array names finfo]] {
        debug-info "    $inf: $finfo($inf)"
    }
    debug-info " $revs revs  $pths files"

    wm title . "$finfo(title) - $g(name) $g(version)"
    return 0
}

###############################################################################
# Set up the display
###############################################################################
proc create-display {} {
    debug-info "create-display"

    global g opts bg tk_version
    global w
    global tmpopts

    # these are the four major areas of the GUI:
    # menubar - the menubar (duh)
    # toolbar - the toolbar (duh, again)
    # client  - the area with the text widgets and the graphical map
    # status us         - a bottom status line

    # this block of destroys is only for stand-alone testing of
    # the GUI code, and can be blown away (or not, if we want to
    # be able to call this routine to recreate the display...)
    catch {
        destroy .menubar
        destroy .toolbar
        destroy .client
        destroy .map
        destroy .status
    }

    # create the top level frames and store them in a global
    # array..
    set w(client) .client
    set w(menubar) .menubar
    set w(toolbar) .toolbar
    set w(status) .status

    # other random windows...
    set w(preferences) .pref
    set w(findDialog) .findDialog
    set w(popupMenu) .popupMenu

    # now, simply build all the pieces
    build-menubar
    build-toolbar
    build-client
    build-status
    build-popupMenu

    frame .separator1 -height 2 -borderwidth 2 -relief groove
    frame .separator2 -height 2 -borderwidth 2 -relief groove

    # ... and fit it all together...
    . configure -menu $w(menubar)
    pack $w(toolbar) -side top -fill x -expand n
    pack .separator1 -side top -fill x -expand n

    pack $w(client) -side top -fill both -expand y
    pack .separator2 -side top -fill x -expand n

    pack $w(status) -side bottom -fill x -expand n

    # apply user preferences by calling the proc that gets
    # called when the user presses "Apply" from the preferences
    # window. That proc uses a global variable named "tmpopts"
    # which should have the values from the dialog. Since we
    # aren't using the dialog, we need to populate this array
    # manually
    foreach key [array names opts] {
        set ::tmpopts($key) $opts($key)
    }
    apply 0

    # Make sure temporary files get deleted
    #bind . <Destroy> {del-tmp}

    # other misc. bindings
    common-navigation $w(LeftText) $w(LeftInfo) $w(LeftCB) $w(RightText) \
      $w(RightInfo) $w(RightCB)

    # normally, keyboard traversal using tab and shift-tab isn't
    # enabled for text widgets, since the default binding for these
    # keys is to actually insert the tab character. Because all of
    # our text widgets are for display only, let's redefine the
    # default binding so the global <Tab> and <Shift-Tab> bindings
    # are used.
    bind Text <Tab> {continue}
    bind Text <Shift-Tab> {continue}

    # if the user toggles scrollbar syncing, we want to make sure
    # they sync up immediately
    trace variable opts(syncscroll) w toggleSyncScroll
    wm deiconify .
    focus -force $w(RightText)
    update idletasks
    # Need this to make the pane-resizing behave
    grid propagate $w(client) f
}

###############################################################################
# when the user changes the "sync scrollbars" option, we want to
# sync up the left scrollbar with the right if they turn the option on
###############################################################################
proc toggleSyncScroll {args} {
    global opts
    global w

    if {$opts(syncscroll) == 1} {
        eval vscroll-sync {{}} 2 [$w(RightText) yview]
    }
}

###############################################################################
# show the popup menu, optionally changing some of the entries based on
# where the user clicked
###############################################################################
proc show-popupMenu {x y} {
    global w
    global g

    set window [winfo containing $x $y]
    if {[winfo class $window] == "Text"} {
        $w(popupMenu) entryconfigure "Find..." -state normal
        $w(popupMenu) entryconfigure "Find Nearest*" -state normal
        $w(popupMenu) entryconfigure "Edit*" -state normal

        if {$window == $w(LeftText) || $window == $w(LeftInfo) || $window == \
          $w(LeftCB)} {
            $w(popupMenu) configure -title "File 1"
            set g(activeWindow) $w(LeftText)
        } else {
            $w(popupMenu) configure -title "File 2"
            set g(activeWindow) $w(RightText)
        }

    } else {
        $w(popupMenu) entryconfigure "Find..." -state disabled
        $w(popupMenu) entryconfigure "Find Nearest*" -state disabled
        $w(popupMenu) entryconfigure "Edit*" -state disabled
    }
    tk_popup $w(popupMenu) $x $y
}

###############################################################################
# build the right-click popup menu
###############################################################################
proc build-popupMenu {} {
    debug-info "build-popupMenu"
    global w g

    # this routine assumes the other windows already exist...
    menu $w(popupMenu)
    foreach win [list LeftText RightText LeftInfo RightInfo LeftCB RightCB \
      mapCanvas] {
        bind $w($win) <3> {show-popupMenu %X %Y}
    }

    set m $w(popupMenu)
    $m add command -label "First Diff" -underline 0 -command [list popupMenu \
      first] -accelerator "f"
    $m add command -label "Previous Diff" -underline 0 -command \
      [list popupMenu previous] -accelerator "p"
    $m add command -label "Center Current Diff" -underline 0 -command \
      [list popupMenu center] -accelerator "c"
    $m add command -label "Next Diff" -underline 0 -command [list popupMenu \
      next] -accelerator "n"
    $m add command -label "Last Diff" -underline 0 -command [list popupMenu \
      last] -accelerator "l"
    $m add separator
    $m add command -label "Find Nearest Diff" -underline 0 -command \
      [list popupMenu nearest] -accelerator "Double-Click"
    $m add separator
    $m add command -label "Find..." -underline 0 -command [list popupMenu find]
    $m add command -label "Edit" -underline 0 -command [list popupMenu edit]
}

###############################################################################
# handle popup menu commands
###############################################################################
proc popupMenu {command args} {
    debug-info "popupMenu ($command $args)"
    global g
    global w

    switch -- $command {
    center {
            center
        }
    edit {
            do-edit
        }
    find {
            do-find
        }
    first {
            move first
        }
    last {
            move last
        }
    next {
            move 1
        }
    previous {
            move -1
        }
    nearest {
            moveNearest $g(activeWindow) xy [winfo pointerx $g(activeWindow)] \
              [winfo pointery $g(activeWindow)]
        }
    }
}

# Resize the text windows relative to each other.  The 8.4 method works
# much better.
proc pane_drag {win x} {
    global w
    global finfo
    global tk_version

    set relX [expr $x - [winfo rootx $win]]
    set maxX [winfo width $win]
    set frac [expr int((double($relX) / $maxX) * 100)]
    if {$tk_version < 8.4} {
      if {$frac < 15} { set frac 15 }
      if {$frac > 85} { set frac 85 }
      #debug-info "frac $frac"
      set L $frac
      set R [expr 100 - $frac]
      .client.leftlabel configure -width [expr $L * 2]
      .client.rightlabel configure -width [expr $R * 2]
    } else {
      if {$frac < 5} { set frac 5 }
      if {$frac > 95} { set frac 95 }
      #debug-info "frac $frac"
      set L $frac
      set R [expr 100 - $frac]
      grid columnconfigure $win 0 -weight $L
      grid columnconfigure $win 2 -weight $R
    }
    #debug-info " new: $L $R"
}

###############################################################################
# build the main client display (the text widgets, scrollbars, that
# sort of fluff)
###############################################################################
proc build-client {} {
    debug-info "build-client"
    global g
    global w
    global opts
    global map
    global tk_version

    frame $w(client) -bd 2 -relief flat

    # set up global variables to reference the widgets, so
    # we don't have to use hardcoded widget paths elsewhere
    # in the code
    #
    # Text  - holds the text of the file
    # Info  - sort-of "invisible" text widget which is kept in sync
    #              with the text widget and holds line numbers
    # CB    - contains changebars or status or something like that...
    # VSB   - vertical scrollbar
    # HSB   - horizontal scrollbar
    # Label - label to hold the name of the file
    set w(LeftText) $w(client).left.text
    set w(LeftInfo) $w(client).left.info
    set w(LeftCB) $w(client).left.changeBars
    set w(LeftVSB) $w(client).left.vsb
    set w(LeftHSB) $w(client).left.hsb
    set w(LeftLabel) $w(client).leftlabel

    set w(RightText) $w(client).right.text
    set w(RightInfo) $w(client).right.info
    set w(RightCB) $w(client).right.changeBars
    set w(RightVSB) $w(client).right.vsb
    set w(RightHSB) $w(client).right.hsb
    set w(RightLabel) $w(client).rightlabel

    set w(BottomText) $w(client).bottomtext

    set w(map) $w(client).map
    set w(mapCanvas) $w(map).canvas

    # these don't need to be global...
    set leftFrame $w(client).left
    set rightFrame $w(client).right

    # we'll create each widget twice; once for the left side
    # and once for the right.
    debug-info "  Assigning labels to headers"
    scan $opts(geometry) "%dx%d" width height
    label $w(LeftLabel) -bd 1 -relief flat -textvariable finfo(lbl,1) -width $width
    label $w(RightLabel) -bd 1 -relief flat -textvariable finfo(lbl,2) -width $width

    # this holds the text widgets and the scrollbars. The reason
    # for the frame is purely for aesthetics. It just looks
    # nicer, IMHO, to "embed" the scrollbars within the text
    # widget
    frame $leftFrame -bd 1 -relief sunken

    frame $rightFrame -bd 1 -relief sunken

    scrollbar $w(LeftHSB) -borderwidth 1 -orient horizontal -command \
      [list $w(LeftText) xview]

    scrollbar $w(RightHSB) -borderwidth 1 -orient horizontal -command \
      [list $w(RightText) xview]

    scrollbar $w(LeftVSB) -borderwidth 1 -orient vertical -command \
      [list $w(LeftText) yview]

    scrollbar $w(RightVSB) -borderwidth 1 -orient vertical -command \
      [list $w(RightText) yview]


    text $w(LeftText) -padx 0 -wrap none -width $width -height $height \
      -borderwidth 0 -setgrid 1 -yscrollcommand [list vscroll-sync \
      "$w(LeftInfo) $w(LeftCB)" 1] -xscrollcommand [list hscroll-sync 1]

    text $w(RightText) -padx 0 -wrap none -width $width -height $height \
      -borderwidth 0 -setgrid 1 -yscrollcommand [list vscroll-sync \
      "$w(RightInfo) $w(RightCB)" 2] -xscrollcommand [list hscroll-sync 2]

    text $w(LeftInfo) -height 0 -padx 0 -width 6 -borderwidth 0 -setgrid 1 \
      -yscrollcommand [list vscroll-sync "$w(LeftCB) $w(LeftText)" 1]

    text $w(RightInfo) -height 0 -padx 0 -width 6 -borderwidth 0 -setgrid 1 \
      -yscrollcommand [list vscroll-sync "$w(RightCB) $w(RightText)" 2]

    # each and every line in a text window will have a corresponding line
    # in this widget. And each line in this widget will be composed of
    # a single character (either "+", "-" or "!" for insertion, deletion
    # or change, respectively
    text $w(LeftCB) -height 0 -padx 0 -highlightthickness 0 -wrap none \
      -foreground white -width 1 -borderwidth 0 -yscrollcommand \
      [list vscroll-sync "$w(LeftInfo) $w(LeftText)" 1]

    text $w(RightCB) -height 0 -padx 0 -highlightthickness 0 -wrap none \
      -background white -foreground white -width 1 -borderwidth 0 \
      -yscrollcommand [list vscroll-sync "$w(RightInfo) $w(RightText)" 2]

    # this widget is the two line display showing the current line, so
    # one can compare character by character if necessary.
    text $w(BottomText) -wrap none -borderwidth 1 -height 2 -width 0

    # this is how we highlight bytes that are different...
    # the bottom window (lineview) uses reverse video to highlight
    # diffs, so we need to figure out what reverse video is, and
    # define the tag appropriately
    eval $w(BottomText) tag configure diff $opts(bytetag)

    # Set up text tags for the 'current diff' (the one chosen by the 'next'
    # and 'prev' buttons) and any ol' diff region.  All diff regions are
    # given the 'diff' tag initially...         As 'next' and 'prev' are \
        pressed,
    # to scroll through the differences, one particular diff region is
    # always chosen as the 'current diff', and is set off from the others
    # via the 'diff' tag -- in particular, so that it's obvious which diffs
    # in the left and right-hand text widgets match.

    foreach widget [list $w(LeftText) $w(LeftInfo) $w(LeftCB) $w(RightText) \
      $w(RightInfo) $w(RightCB)] {
        eval "$widget configure $opts(textopt)"
        foreach tag {difftag currtag inlinetag deltag instag chgtag \
          overlaptag + - ! ?} {
            eval "$widget tag configure $tag $opts($tag)"
        }
    }

    # adjust the tag priorities a bit...
    foreach window [list LeftText RightText LeftCB RightCB LeftInfo RightInfo] {
        $w($window) tag raise deltag currtag
        $w($window) tag raise chgtag currtag
        $w($window) tag raise instag currtag
        $w($window) tag raise currtag difftag
        $w($window) tag raise inlinetag
    }

    # these tags are specific to change bars
    foreach widget [list $w(LeftCB) $w(RightCB)] {
        eval "$widget tag configure + $opts(+)"
        eval "$widget tag configure - $opts(-)"
        eval "$widget tag configure ! $opts(!)"
        eval "$widget tag configure ? $opts(?)"
    }

    # build the map...
    # we want the map to be the same width as a scrollbar, so we'll
    # steal some information from one of the scrollbars we just
    # created...
    set cwidth [winfo reqwidth $w(LeftVSB)]
    set ht [$w(LeftVSB) cget -highlightthickness]
    set cwidth [expr {$cwidth -($ht*2)}]
    set color [$w(LeftVSB) cget -troughcolor]

    set map [frame $w(client).map -bd 1 -relief sunken -takefocus 0 \
      -highlightthickness 0]

    # now for the real map...
    image create photo map

    canvas $w(mapCanvas) -width [expr {$cwidth + 1}] \
      -yscrollcommand map-resize -background $color -borderwidth 0 \
      -relief sunken -highlightthickness 0
    $w(mapCanvas) create image 1 1 -image map -anchor nw
    pack $w(mapCanvas) -side top -fill both -expand y

    # I'm not too pleased with these bindings -- it results in a rather
    # jerky, cpu-intensive maneuver since with each move of the mouse
    # we are finding and tagging the nearest diff. But, what *should*
    # it do?
    #
    # I think what I *want* it to do is update the combobox and status
    # bar so the user can see where in the scheme of things they are,
    # but not actually select anything until they release the mouse.
    bind $w(mapCanvas) <ButtonPress-1> [list handleMapEvent B1-Press %y]
    bind $w(mapCanvas) <Button1-Motion> [list handleMapEvent B1-Motion %y]
    bind $w(mapCanvas) <ButtonRelease-1> [list handleMapEvent B1-Release %y]

    # this is a grip for resizing the sides relative to each other.
    button $w(client).grip -borderwidth 3 -relief raised \
      -cursor sb_h_double_arrow -image resize
    bind $w(client).grip <B1-Motion> {pane_drag $w(client) %X}

    # use grid to manage the widgets in the left side frame
    grid $w(LeftVSB) -row 0 -column 0 -sticky ns
    grid $w(LeftInfo) -row 0 -column 1 -sticky nsew
    grid $w(LeftCB) -row 0 -column 2 -sticky ns
    grid $w(LeftText) -row 0 -column 3 -sticky nsew
    grid $w(LeftHSB) -row 1 -column 1 -sticky ew -columnspan 3

    grid rowconfigure $leftFrame 0 -weight 1
    grid rowconfigure $leftFrame 1 -weight 0

    grid columnconfigure $leftFrame 0 -weight 0
    grid columnconfigure $leftFrame 1 -weight 0
    grid columnconfigure $leftFrame 2 -weight 0
    grid columnconfigure $leftFrame 3 -weight 1

    # likewise for the right...
    grid $w(RightVSB) -row 0 -column 3 -sticky ns
    grid $w(RightInfo) -row 0 -column 0 -sticky nsew
    grid $w(RightCB) -row 0 -column 1 -sticky ns
    grid $w(RightText) -row 0 -column 2 -sticky nsew
    grid $w(RightHSB) -row 1 -column 0 -sticky ew -columnspan 3

    grid rowconfigure $rightFrame 0 -weight 1
    grid rowconfigure $rightFrame 1 -weight 0

    grid columnconfigure $rightFrame 0 -weight 0
    grid columnconfigure $rightFrame 1 -weight 0
    grid columnconfigure $rightFrame 2 -weight 1
    grid columnconfigure $rightFrame 3 -weight 0

    # use grid to manage the labels, frames and map. We're going to
    # toss in an extra row just for the benefit of our dummy frame.
    # the intent is that the dummy frame will match the height of
    # the horizontal scrollbars so the map stops at the right place...
    grid $w(LeftLabel) -row 0 -column 0 -sticky ew
    grid $w(RightLabel) -row 0 -column 2 -sticky ew
    grid $leftFrame -row 1 -column 0 -sticky nsew -rowspan 2
    grid $map -row 1 -column 1 -stick ns
    grid $w(client).grip -row 2 -column 1
    grid $rightFrame -row 1 -column 2 -sticky nsew -rowspan 2

    grid rowconfigure $w(client) 0 -weight 0
    grid rowconfigure $w(client) 1 -weight 1
    grid rowconfigure $w(client) 2 -weight 0
    grid rowconfigure $w(client) 3 -weight 0

    if {$tk_version < 8.4} {
      grid columnconfigure $w(client) 0 -weight 1
      grid columnconfigure $w(client) 2 -weight 1
    } else {
      grid columnconfigure $w(client) 0 -weight 100 -uniform a
      grid columnconfigure $w(client) 2 -weight 100 -uniform a
    }
    grid columnconfigure $w(client) 1 -weight 0

    # this adjusts the variable g(activeWindow) to be whatever text
    # widget has the focus...
    bind $w(LeftText) <1> {set g(activeWindow) $w(LeftText)}
    bind $w(RightText) <1> {set g(activeWindow) $w(RightText)}

    set g(activeWindow) $w(LeftText) ;# establish a default

    rename $w(RightText) $w(RightText)_
    rename $w(LeftText) $w(LeftText)_

    proc $w(RightText) {command args} $::text_widget_proc
    proc $w(LeftText) {command args} $::text_widget_proc
}

###############################################################################
# Functionality: Inline diffs
# Athr: Michael D. Beynon : mdb - beynon@yahoo.com
# Date: 04/08/2003 : mdb - Added inline character diffs.
#       04/16/2003 : mdb - Rewrote longest-common-substring to be faster.
#                        - Added byte-by-byte algorithm.
#
# the recursive version is derived from the Ratcliff/Obershelp pattern
# recognition algorithm (Dr Dobbs July 1988), where we search for a
# longest common substring between two strings.  This match is used as
# an archor, around which we recursively do the same for the two left
# and two right remaining pieces (omitting the anchor).  This
# precisely determines the location of the intraline tags.
#################################################################################
proc longest-common-substring {s1 off1 len1 s2 off2 len2 lcsoff1_ref \
  lcsoff2_ref} {
    upvar $lcsoff1_ref lcsoff1
    upvar $lcsoff2_ref lcsoff2
    set snippet ""

    set snippetlen 0
    set longestlen 0

    # extract just the search regions for efficiency in string searching
    set s1 [string range $s1 $off1 [expr $off1+$len1-1]]
    set s2 [string range $s2 $off2 [expr $off2+$len2-1]]

    set j 0

    while {1} {
        # increase size of matching snippet
        while {$snippetlen < $len2-$j} {
            set tmp "$snippet[string index $s2 [expr $j+$snippetlen]]"
            if {[string first $tmp $s1] == -1} {
                break
            }
            set snippet $tmp
            incr snippetlen
        }
        if {$snippetlen == 0} {
            # nothing starting at this position
            incr j
            if {$snippetlen >= $len2-$j} {
                break
            }
        } else {
            set tmpoff [string first $snippet $s1]
            if {$tmpoff != -1 && $snippetlen > $longestlen} {
                # new longest?
                set longest $snippet
                set longestlen $snippetlen
                set lcsoff1 [expr $off1+$tmpoff]
                set lcsoff2 [expr $off2+$j]
            }
            # drop 1st char of prefix, but keep size the same as longest
            if {$snippetlen >= $len2-$j} {
                break
            }
            set snippet "[string range $snippet 1 end][string index $s2 \
              [expr $j+$snippetlen]]"
            incr j
        }
    }
    return $longestlen
}

proc fid-ratcliff-aux {pos l1 l2 s1 off1 len1 s2 off2 len2} {
    global g

    if {$len1 <= 0 || $len2 <= 0} {
        if {$len1 == 0} {
            set g(scrinline,$pos,$g(scrinline,$pos)) [list r $l2 $off2 \
              [expr $off2+$len2]]
            incr g(scrinline,$pos)
        } elseif {$len2 == 0} {
            set g(scrinline,$pos,$g(scrinline,$pos)) [list l $l1 $off1 \
              [expr $off1+$len1]]
            incr g(scrinline,$pos)
        }
        return 0
    }
    set cnt 0
    set lcsoff1 -1
    set lcsoff2 -1

    set ret [longest-common-substring $s1 $off1 $len1 $s2 $off2 $len2 lcsoff1 \
      lcsoff2]


    if {$ret > 0} {
        set rightoff1 [expr $lcsoff1+$ret]
        set rightoff2 [expr $lcsoff2+$ret]

        incr cnt [expr 2*$ret]
        if {$lcsoff1 > $off1 || $lcsoff2 > $off2} {
            # left
            incr cnt [fid-ratcliff-aux $pos $l1 $l2 $s1 $off1 \
              [expr $lcsoff1-$off1] $s2 $off2 [expr $lcsoff2-$off2]]

        }
        if {$rightoff1<$off1+$len1 || $rightoff2<$off2+$len2} {
            # right
            incr cnt [fid-ratcliff-aux $pos $l1 $l2 $s1 $rightoff1 \
              [expr $off1+$len1-$rightoff1] $s2 $rightoff2 \
              [expr $off2+$len2-$rightoff2]]
        }
    } else {
        set g(scrinline,$pos,$g(scrinline,$pos)) [list r $l2 $off2 \
          [expr $off2+$len2]]
        incr g(scrinline,$pos)
        set g(scrinline,$pos,$g(scrinline,$pos)) [list l $l1 $off1 \
          [expr $off1+$len1]]
        incr g(scrinline,$pos)
    }
    return $cnt
}

proc find-inline-diff-ratcliff {pos l1 l2 s1 s2} {
    global g

    set len1 [string length $s1]
    set len2 [string length $s2]
    if {$len1 == 0 || $len2 == 0} {
        return 0
    }
    return [fid-ratcliff-aux $pos $l1 $l2 $s1 0 $len1 $s2 0 $len2]
}

proc find-inline-diff-byte {pos l1 l2 s1 s2} {
    global g

    set len1 [string length $s1]
    set len2 [string length $s2]
    if {$len1 == 0 || $len2 == 0} {
        return 0
    }

    set cnt 0

    set lenmin [min $len1 $len2]
    set size 0
    for {set i 0} {$i < $lenmin} {incr i} {
        if {$size > 0} {
            # in a diff section
            if {[string index $s1 $i] == [string index $s2 $i]} {
                # end of diff region
                set g(scrinline,$pos,$g(scrinline,$pos)) [list r $l2 \
                  [expr $i-$size] $i]
                incr g(scrinline,$pos)
                set g(scrinline,$pos,$g(scrinline,$pos)) [list l $l1 \
                  [expr $i-$size] $i]
                incr g(scrinline,$pos)
                set size 0
                incr cnt
            } else {
                incr size
            }
        } else {
            if {[string index $s1 $i] != [string index $s2 $i]} {
                set size 1
            }
        }
    }
    if {$size > 0} {
        # end of diff region
        set g(scrinline,$pos,$g(scrinline,$pos)) [list r $l2 [expr $i-$size] \
          $len2]
        incr g(scrinline,$pos)
        set g(scrinline,$pos,$g(scrinline,$pos)) [list l $l1 [expr $i-$size] \
          $len1]
        incr g(scrinline,$pos)
        incr cnt
    }
    return $cnt
}

###############################################################################
# the following code is used as the replacement body for the left and
# right widget procs. The purpose is to catch when the insertion point
# changes so we can update the line comparison window
###############################################################################

set text_widget_proc {
    global w
    set real "[lindex [info level [info level]] 0]_"
    set result [eval $real $command $args]
    if {$command == "mark"} {
        if {[lindex $args 0] == "set" && [lindex $args 1] == "insert"} {
            set i [lindex $args 2]
            set i0 "$i linestart"
            set i1 "$i lineend"
            set left [$w(LeftText)_ get $i0 $i1]
            set right [$w(RightText)_ get $i0 $i1]
            $w(BottomText) delete 1.0 end
            $w(BottomText) insert end "< $left\n> $right"
            # find characters that are different, and underline them
            if {$left != $right} {
                set left [split $left {}]
                set right [split $right {}]
                # n.b. we set c to an offset equal to whatever we have
                # prepended to the data...
                set c 2
                foreach l $left r $right {
                    if {[string compare $l $r] != 0} {
                        $w(BottomText) tag add diff 1.$c "1.$c+1c"
                        $w(BottomText) tag add diff 2.$c "2.$c+1c"
                    }
                    incr c
                }
                $w(BottomText) tag remove diff "1.0 lineend"
                $w(BottomText) tag remove diff "2.0 lineend"
            }
        }
    }
    return $result
}

###############################################################################
# create (if necessary) and show the find dialog
###############################################################################
proc show-find {} {
    debug-info "show-find"
    global w g
    global tcl_platform

    if {![winfo exists $w(findDialog)]} {
        toplevel $w(findDialog)
        wm group $w(findDialog) .
        wm transient $w(findDialog) .
        wm title $w(findDialog) "$g(name) Find"

        if {$g(windowingSystem) == "aqua"} {
            setAquaDialogStyle $w(findDialog)
        }

        # we don't want the window to be deleted, just hidden from view
        wm protocol $w(findDialog) WM_DELETE_WINDOW [list wm withdraw \
          $w(findDialog)]

        wm withdraw $w(findDialog)
        update idletasks

        frame $w(findDialog).content -bd 2 -relief groove
        pack $w(findDialog).content -side top -fill both -expand y -padx 0 \
          -pady 5

        frame $w(findDialog).buttons
        pack $w(findDialog).buttons -side bottom -fill x -expand n

        button $w(findDialog).buttons.doit -text "Find Next" -command do-find
        button $w(findDialog).buttons.dismiss -text "Dismiss" -command \
          "wm withdraw $w(findDialog)"
        pack $w(findDialog).buttons.dismiss -side right -pady 5 -padx 0
        pack $w(findDialog).buttons.doit -side right -pady 5 -padx 1

        set ff $w(findDialog).content.findFrame
        frame $ff -height 100 -bd 2 -relief flat
        pack $ff -side top -fill x -expand n -padx 0 -pady 5

        label $ff.label -text "Find what:" -underline 2

        entry $ff.entry -textvariable g(findString)

        checkbutton $ff.searchCase -text "Ignore Case" -offvalue 0 -onvalue 1 \
          -indicatoron true -variable g(findIgnoreCase)

        grid $ff.label -row 0 -column 0 -sticky e
        grid $ff.entry -row 0 -column 1 -sticky ew
        grid $ff.searchCase -row 0 -column 2 -sticky w
        grid columnconfigure $ff 0 -weight 0
        grid columnconfigure $ff 1 -weight 1
        grid columnconfigure $ff 2 -weight 0

        # we need this in other places...
        set w(findEntry) $ff.entry

        bind $ff.entry <Return> do-find

        set of $w(findDialog).content.optionsFrame
        frame $of -bd 2 -relief flat
        pack $of -side top -fill y -expand y -padx 10 -pady 10

        label $of.directionLabel -text "Search Direction:" -anchor e
        radiobutton $of.directionForward -indicatoron true -text "Down" \
          -value "-forward" -variable g(findDirection)
        radiobutton $of.directionBackward -text "Up" -value "-backward" \
          -indicatoron true -variable g(findDirection)


        label $of.windowLabel -text "Window:" -anchor e
        radiobutton $of.windowLeft -indicatoron true -text "Left" \
          -value $w(LeftText) -variable g(activeWindow)
        radiobutton $of.windowRight -indicatoron true -text "Right" \
          -value $w(RightText) -variable g(activeWindow)


        label $of.searchLabel -text "Search Type:" -anchor e
        radiobutton $of.searchExact -indicatoron true -text "Exact" \
          -value "-exact" -variable g(findType)
        radiobutton $of.searchRegexp -text "Regexp" -value "-regexp" \
          -indicatoron true -variable g(findType)

        grid $of.directionLabel -row 1 -column 0 -sticky w
        grid $of.directionForward -row 1 -column 1 -sticky w
        grid $of.directionBackward -row 1 -column 2 -sticky w

        grid $of.windowLabel -row 0 -column 0 -sticky w
        grid $of.windowLeft -row 0 -column 1 -sticky w
        grid $of.windowRight -row 0 -column 2 -sticky w

        grid $of.searchLabel -row 2 -column 0 -sticky w
        grid $of.searchExact -row 2 -column 1 -sticky w
        grid $of.searchRegexp -row 2 -column 2 -sticky w

        grid columnconfigure $of 0 -weight 0
        grid columnconfigure $of 1 -weight 0
        grid columnconfigure $of 2 -weight 1

        set g(findDirection) "-forward"
        set g(findType) "-exact"
        set g(findIgnoreCase) 1
        set g(lastSearch) ""
        if {$g(activeWindow) == ""} {
            set g(activeWindow) [focus]
            if {$g(activeWindow) != $w(LeftText) && $g(activeWindow) != \
              $w(RightText)} {
                set g(activeWindow) $w(LeftText)
            }
        }
    }

    centerWindow $w(findDialog)
    wm deiconify $w(findDialog)
    raise $w(findDialog)
    after idle focus $w(findEntry)
}


###############################################################################
# do the "Edit->Copy" functionality, by copying the current selection
# to the clipboard
###############################################################################
proc do-copy {} {
    clipboard clear -displayof .
    # figure out which window has the selection...
    catch {
        clipboard append [selection get -displayof .]
    }
}

###############################################################################
# search for the text in the find dialog
###############################################################################
proc do-find {} {
    global g
    global w

    if {![winfo exists $w(findDialog)] || ![winfo ismapped $w(findDialog)]} {
        show-find
        return
    }

    set win $g(activeWindow)
    if {$win == ""} {
        set win $w(LeftText)
    }
    if {$g(lastSearch) != ""} {
        if {$g(findDirection) == "-forward"} {
            set start [$win index "insert +1c"]
        } else {
            set start insert
        }
    } else {
        set start 1.0
    }

    if {$g(findIgnoreCase)} {
        set result [$win search $g(findDirection) $g(findType) -nocase \
          -- $g(findString) $start]
    } else {
        set result [$win search $g(findDirection) $g(findType) \
          -- $g(findString) $start]
    }
    if {[string length $result] > 0} {
        # if this is a regular expression search, get the whole line and try
        # to figure out exactly what matched; otherwise we know we must
        # have matched the whole string...
        if {$g(findType) == "-regexp"} {
            set line [$win get $result "$result lineend"]
            regexp $g(findString) $line matchVar
            set length [string length $matchVar]
        } else {
            set length [string length $g(findString)]
        }
        set g(lastSearch) $result
        $win mark set insert $result
        $win tag remove sel 1.0 end
        $win tag add sel $result "$result + ${length}c"
        $win see $result
        focus $win
        # should I somehow snap to the nearest diff? Probably not...
    } else {
        bell

    }
}

###############################################################################
# Build the menu bar
###############################################################################
proc build-menubar {} {
    debug-info "build-menubar"
    global g
    global opts
    global w

    menu $w(menubar)

    # this is just temporary shorthand ...
    set menubar $w(menubar)


    # First, the menu buttons...
    set fileMenu $w(menubar).file
    set viewMenu $w(menubar).view
    set helpMenu $w(menubar).help
    set editMenu $w(menubar).edit
    set mergeMenu $w(menubar).window
    set markMenu $w(menubar).marks

    $w(menubar) add cascade -label "File" -menu $fileMenu -underline 0
    $w(menubar) add cascade -label "Edit" -menu $editMenu -underline 0
    $w(menubar) add cascade -label "View" -menu $viewMenu -underline 0
    $w(menubar) add cascade -label "Mark" -menu $markMenu -underline 3
    $w(menubar) add cascade -label "Merge" -menu $mergeMenu -underline 0
    $w(menubar) add cascade -label "Help" -menu $helpMenu -underline 0

    # these, however, are used in other places..
    set w(fileMenu) $fileMenu
    set w(viewMenu) $viewMenu
    set w(helpMenu) $helpMenu
    set w(editMenu) $editMenu
    set w(mergeMenu) $mergeMenu
    set w(markMenu) $markMenu

    # Now, the menus...

    # Mark menu...
    menu $markMenu
    $markMenu add command -label "Mark Current Diff" -command [list diffmark \
      mark] -underline 0
    $markMenu add command -label "Clear Current Diff Mark" -command \
      [list diffmark clear] -underline 0

    set "g(tooltip,Mark Current Diff)" "Create a marker for the current \
      difference record"
    set "g(tooltip,Clear Current Diff Mark)" "Clear the marker for the \
      current difference record"

    # File menu...
    menu $fileMenu
    $fileMenu add command -label "New..." -underline 0 -command {do-new-diff}
    $fileMenu add separator
    $fileMenu add command -label "Recompute Diffs" -underline 0 \
      -accelerator r -command recompute-diff
    $fileMenu add command -label "Write Report..." -command \
      [list write-report popup] -underline 0
    $fileMenu add separator
    $fileMenu add command -label "Exit" -underline 1 -accelerator q \
      -command do-exit

    # Edit menu...  If you change, add or remove labels, be sure and
    # update the tooltips.
    menu $editMenu
    $editMenu add command -label "Copy" -underline 0 -command do-copy
    $editMenu add separator
    $editMenu add command -label "Find..." -underline 0 -command show-find
    $editMenu add separator
    $editMenu add command -label "Edit File 1" -command {
        set g(activeWindow) $w(LeftText)
        do-edit
    } -underline 10
    $editMenu add command -label "Edit File 2" -command {
        set g(activeWindow) $w(RightText)
        do-edit
    } -underline 10
    $editMenu add separator
    $editMenu add command -label "Preferences..." -underline 0 \
      -command customize

    set "g(tooltip,Copy)" "Copy the currently selected text to the clipboard"
    set "g(tooltip,Find...)" "Pop up a dialog to search for a string within \
      either file"
    set "g(tooltip,Edit File 1)" "Launch an editor on the file on the left \
      side of the window"
    set "g(tooltip,Edit File 2)" "Launch an editor on the file on the right \
      side of the window"
    set "g(tooltip,Preferences...)" "Pop up a window to customize $g(name)"

    # View menu...  If you change, add or remove labels, be sure and
    # update the tooltips.
    menu $viewMenu
    $viewMenu add checkbutton -label "Ignore White Spaces" -underline 8 \
      -selectcolor $w(selcolor) -variable opts(ignoreblanks) \
      -command do-show-ignoreblanks

    $viewMenu add checkbutton -label "Show Line Numbers" -underline 12 \
      -selectcolor $w(selcolor) -variable opts(showln) \
      -command do-show-linenumbers

    $viewMenu add checkbutton -label "Show Change Bars" -underline 12 \
      -selectcolor $w(selcolor) -variable opts(showcbs) \
      -command do-show-changebars

    $viewMenu add checkbutton -label "Show Diff Map" -underline 5 \
      -selectcolor $w(selcolor) -variable opts(showmap) -command do-show-map

    $viewMenu add checkbutton -label "Show Line Comparison Window" \
      -underline 11 -selectcolor $w(selcolor) -variable opts(showlineview) \
      -command do-show-lineview

    $viewMenu add checkbutton -label "Show Inline Comparison (byte)" \
      -selectcolor $w(selcolor) -variable opts(showinline1) \
      -command do-show-inline1

    $viewMenu add checkbutton -label "Show Inline Comparison (recursive)" \
      -selectcolor $w(selcolor) -variable opts(showinline2) \
      -command do-show-inline2

    $viewMenu add separator

    $viewMenu add checkbutton -label "Synchronize Scrollbars" -underline 0 \
      -selectcolor $w(selcolor) -variable opts(syncscroll)
    $viewMenu add checkbutton -label "Auto Center" -underline 0 \
      -selectcolor $w(selcolor) -variable opts(autocenter) -command {if \
      {$opts(autocenter)} {center}}
    $viewMenu add checkbutton -label "Auto Select" -underline 1 \
      -selectcolor $w(selcolor) -variable opts(autoselect)

    $viewMenu add separator

    $viewMenu add command -label "First Diff" -underline 0 -command \
      {move first} -accelerator "F"
    $viewMenu add command -label "Previous Diff" -underline 0 -command {move \
      -1} -accelerator "P"
    $viewMenu add command -label "Center Current Diff" -underline 0 \
      -command {center} -accelerator "C"
    $viewMenu add command -label "Next Diff" -underline 0 -command {move 1} \
      -accelerator "N"
    $viewMenu add command -label "Last Diff" -underline 0 -command \
      {move last} -accelerator "L"

    set "g(tooltip,Show Change Bars)" "If set, show the changebar column for \
       each line of each file"
    set "g(tooltip,Show Line Numbers)" "If set, show line numbers beside each \
       line of each file"
    set "g(tooltip,Synchronize Scrollbars)" "If set, scrolling either window \
       will scroll both windows"
    set "g(tooltip,Diff Map)" "If set, display the graphical \"Difference \
      Map\" in the center of the display"
    set "g(tooltip,Show Line Comparison Window)" "If set, display the window \
       with byte-by-byte differences"
    set "g(tooltip,Show Inline Comparison (byte))" "If set, display inline \
      byte-by-byte differences"
    set "g(tooltip,Show Inline Comparison (recursive))" "If set, display \
      inline differences based on recursive matching regions"
    set "g(tooltip,Auto Select)" "If set, automatically selects the nearest \
       diff record while scrolling"
    set "g(tooltip,Auto Center)" "If set, moving to another diff record will \
       center the diff on the screen"
    set "g(tooltip,Center Current Diff)" "Center the display around the \
      current diff record"
    set "g(tooltip,First Diff)" "Go to the first difference"
    set "g(tooltip,Last Diff)" "Go to the last difference"
    set "g(tooltip,Previous Diff)" "Go to the diff record just prior to the \
       current diff record"
    set "g(tooltip,Next Diff)" "Go to the diff record just after the current \
       diff record"
    set "g(tooltip,Ignore White Spaces)" "If set, changes in whitespaces are \
       ignored"

    # Merge menu. If you change, add or remove labels, be sure and
    # update the tooltips.
    menu $mergeMenu
    $mergeMenu add checkbutton -label "Show Merge Window" -underline 9 \
      -selectcolor $w(selcolor) -variable g(showmerge) -command "do-show-merge 1"
    $mergeMenu add command -label "Write Merge File..." -underline 6 \
      -command popup-merge
    set "g(tooltip,Show Merge Window)" "Pops up a window showing the current \
       merge results"
    set "g(tooltip,Write Merge File)" "Write the merge file to disk. You will \
       be prompted for a filename"

    # Help menu. If you change, add or remove labels, be sure and
    # update the tooltips.
    menu $helpMenu
    $helpMenu add command -label "On GUI" -underline 3 -command do-help
    $helpMenu add command -label "On Command Line" -underline 3 \
      -command "do-usage gui"
    $helpMenu add command -label "On Preferences" -underline 3 \
      -command do-help-preferences
    $helpMenu add separator
    $helpMenu add command -label "About $g(name)" -underline 0 -command do-about

    bind $fileMenu <<MenuSelect>> {showTooltip menu %W}
    bind $editMenu <<MenuSelect>> {showTooltip menu %W}
    bind $viewMenu <<MenuSelect>> {showTooltip menu %W}
    bind $markMenu <<MenuSelect>> {showTooltip menu %W}
    bind $mergeMenu <<MenuSelect>> {showTooltip menu %W}
    bind $helpMenu <<MenuSelect>> {showTooltip menu %W}

    set "g(tooltip,On Preferences)" "Show help on the user-settable preferences"
    set "g(tooltip,On GUI)" "Show help on how to use the Graphical User \
      Interface"
    set "g(tooltip,On Command Line)" "Show help on the command line arguments"
    set "g(tooltip,About $g(name))" "Show information about this application"
}

###############################################################################
# Show explanation of item in the status bar at the bottom.
# Now used only for menu items
###############################################################################
proc showTooltip {which w} {
    global tooltip
    global g
    switch -- $which {
    menu {
            if {[catch {$w entrycget active -label} label]} {
                set label ""
            }
            if {[info exists g(tooltip,$label)]} {
                set g(statusInfo) $g(tooltip,$label)
            } else {
                set g(statusInfo) $label
            }
            update idletasks
        }
    button {
            if {[info exists g(tooltip,$w)]} {
                set g(statusInfo) $g(tooltip,$w)
            } else {
                set g(statusInfo) ""
            }
            update idletasks
        }
    }
}

###############################################################################
# Build the toolbar, in text or image mode
###############################################################################
proc build-toolbar {} {
    debug-info "build-toolbar"
    global w g
    global opts

    frame $w(toolbar) -bd 0

    set toolbar $w(toolbar)

    # these are used in other places..
    set w(combo) $toolbar.combo
    set w(rediff_im) $toolbar.rediff_im
    set w(rediff_tx) $toolbar.rediff_tx
    set w(find_im) $toolbar.find_im
    set w(find_tx) $toolbar.find_tx
    set w(mergeChoiceLabel) $toolbar.mergechoicelbl
    set w(mergeChoice1_im) $toolbar.m1_im
    set w(mergeChoice1_tx) $toolbar.m1_tx
    set w(mergeChoice2_im) $toolbar.m2_im
    set w(mergeChoice2_tx) $toolbar.m2_tx
    set w(mergeChoice12_im) $toolbar.m12_im
    set w(mergeChoice12_tx) $toolbar.m12_tx
    set w(mergeChoice21_im) $toolbar.m21_im
    set w(mergeChoice21_tx) $toolbar.m21_tx
    set w(diffNavLabel) $toolbar.diffnavlbl
    set w(prevDiff_im) $toolbar.prev_im
    set w(prevDiff_tx) $toolbar.prev_tx
    set w(firstDiff_im) $toolbar.first_im
    set w(firstDiff_tx) $toolbar.first_tx
    set w(lastDiff_im) $toolbar.last_im
    set w(lastDiff_tx) $toolbar.last_tx
    set w(nextDiff_im) $toolbar.next_im
    set w(nextDiff_tx) $toolbar.next_tx
    set w(centerDiffs_im) $toolbar.center_im
    set w(centerDiffs_tx) $toolbar.center_tx
    set w(markLabel) $toolbar.bkmklbl
    set w(markSet_im) $toolbar.bkmkset_im
    set w(markSet_tx) $toolbar.bkmkset_tx
    set w(markClear_im) $toolbar.bkmkclear_im
    set w(markClear_tx) $toolbar.bkmkclear_tx

    # separators
    toolsep $toolbar.sep1
    toolsep $toolbar.sep2
    toolsep $toolbar.sep3
    toolsep $toolbar.sep4
    toolsep $toolbar.sep5
    toolsep $toolbar.sep6

    # The combo box
    ::combobox::combobox $toolbar.combo -borderwidth 1 -editable false \
      -command moveTo -width 20

    # rediff...
    toolbutton $toolbar.rediff_im -image rediffImage -command recompute-diff \
      -bd 1
    toolbutton $toolbar.rediff_tx -text "Rediff" -command recompute-diff \
      -bd 1 -pady 1

    # find...
    toolbutton $toolbar.find_im -image findImage -command do-find -bd 1
    toolbutton $toolbar.find_tx -text "Find" -command do-find -bd 1 -pady 1

    # navigation widgets
    label $toolbar.diffnavlbl -text "Diff:" -pady 0 -bd 2 -relief groove

    toolbutton $toolbar.prev_im -image prevDiffImage -command [list move -1] \
      -bd 1
    toolbutton $toolbar.prev_tx -text "Prev" -command [list move -1] -bd 1 \
      -pady 1

    toolbutton $toolbar.next_im -image nextDiffImage -command [list move 1] \
      -bd 1
    toolbutton $toolbar.next_tx -text "Next" -command [list move 1] -bd 1 \
      -pady 1

    toolbutton $toolbar.first_im -image firstDiffImage -command [list move \
      first] -bd 1
    toolbutton $toolbar.first_tx -text "First" -command [list move first] \
      -bd 1 -pady 1

    toolbutton $toolbar.last_im -image lastDiffImage -command [list move \
      last] -bd 1
    toolbutton $toolbar.last_tx -text "Last" -command [list move last] -bd 1 \
      -pady 1

    toolbutton $toolbar.center_im -image centerDiffsImage -command center -bd 1
    toolbutton $toolbar.center_tx -text "Center" -command center -bd 1 -pady 1

    # the merge widgets
    label $toolbar.mergechoicelbl -text "Merge:" -pady 0 -bd 2 -relief groove

    radiobutton $toolbar.m2_im -borderwidth 1 -indicatoron false \
      -image mergeChoice2Image -value 2 -variable g(toggle) -command \
      [list do-merge-choice 2] -takefocus 0
    radiobutton $toolbar.m2_tx -borderwidth 1 -indicatoron true -text "R" \
      -value 2 -variable g(toggle) -command [list do-merge-choice 2] \
      -takefocus 0

    radiobutton $toolbar.m1_im -borderwidth 1 -indicatoron false \
      -image mergeChoice1Image -value 1 -variable g(toggle) -command \
      [list do-merge-choice 1] -takefocus 0
    radiobutton $toolbar.m1_tx -borderwidth 1 -indicatoron true -text "L" \
      -value 1 -variable g(toggle) -command [list do-merge-choice 1] \
      -takefocus 0

    radiobutton $toolbar.m12_im -borderwidth 1 -indicatoron false \
      -image mergeChoice12Image -value 12 -variable g(toggle) -command \
      [list do-merge-choice 12] -takefocus 0
    radiobutton $toolbar.m12_tx -borderwidth 1 -indicatoron true -text "LR" \
      -value 12 -variable g(toggle) -command [list do-merge-choice 12] \
      -takefocus 0

    radiobutton $toolbar.m21_im -borderwidth 1 -indicatoron false \
      -image mergeChoice21Image -value 21 -variable g(toggle) -command \
      [list do-merge-choice 21] -takefocus 0
    radiobutton $toolbar.m21_tx -borderwidth 1 -indicatoron true -text "RL" \
      -value 21 -variable g(toggle) -command [list do-merge-choice 21] \
      -takefocus 0

    # The bookmarks
    label $toolbar.bkmklbl -text "Mark:" -pady 0 -bd 2 -relief groove

    toolbutton $toolbar.bkmkset_im -image markSetImage -command \
      [list diffmark mark] -bd 1
    toolbutton $toolbar.bkmkset_tx -text "Set" -command [list diffmark mark] \
      -bd 1 -pady 1

    toolbutton $toolbar.bkmkclear_im -image markClearImage -command \
      [list diffmark clear] -bd 1
    toolbutton $toolbar.bkmkclear_tx -text "Clear" -command [list diffmark \
      clear] -bd 1 -pady 1

    set_tooltips $w(find_im) {"Pop up a dialog to search for a string within \
      either file"}
    set_tooltips $w(find_tx) {"Pop up a dialog to search for a string within \
      either file"}
    set_tooltips $w(rediff_im) {"Recompute and redisplay the difference \
      records"}
    set_tooltips $w(rediff_tx) {"Recompute and redisplay the difference \
      records"}
    set_tooltips $w(mergeChoice12_im) {"select the diff on the left then \
      right for  merging"}
    set_tooltips $w(mergeChoice12_tx) {"select the diff on the left then \
      right for  merging"}
    set_tooltips $w(mergeChoice1_im) {"select the diff on the left for merging"}
    set_tooltips $w(mergeChoice1_tx) {"select the diff on the left for merging"}
    set_tooltips $w(mergeChoice2_im) {"select the diff on the right for \
      merging"}
    set_tooltips $w(mergeChoice2_tx) {"select the diff on the right for \
      merging"}
    set_tooltips $w(mergeChoice21_im) {"select the diff on the right then \
      left for  merging"}
    set_tooltips $w(mergeChoice21_tx) {"select the diff on the right then \
      left for  merging"}
    set_tooltips $w(prevDiff_im) {"Previous Diff"}
    set_tooltips $w(prevDiff_tx) {"Previous Diff"}
    set_tooltips $w(nextDiff_im) {"Next Diff"}
    set_tooltips $w(nextDiff_tx) {"Next Diff"}
    set_tooltips $w(firstDiff_im) {"First Diff"}
    set_tooltips $w(firstDiff_tx) {"First Diff"}
    set_tooltips $w(lastDiff_im) {"Last Diff"}
    set_tooltips $w(lastDiff_tx) {"Last Diff"}
    set_tooltips $w(markSet_im) {"Mark current diff"}
    set_tooltips $w(markSet_tx) {"Mark current diff"}
    set_tooltips $w(markClear_im) {"Clear current diff mark"}
    set_tooltips $w(markClear_tx) {"Clear current diff mark"}
    set_tooltips $w(centerDiffs_im) {"Center Current Diff"}
    set_tooltips $w(centerDiffs_tx) {"Center Current Diff"}

    pack-toolbuttons $toolbar
}

proc pack-toolbuttons {toolbar} {
    #debug-info "pack-toolbuttons ($toolbar)"
    global opts

    if {$opts(toolbarIcons)} {
        set bp "im"
    } else {
        set bp "tx"
    }

    pack $toolbar.combo -side left -padx 2
    pack $toolbar.sep1 -side left -fill y -pady 2 -padx 2
    pack $toolbar.rediff_$bp -side left -padx 2
    pack $toolbar.find_$bp -side left -padx 2
    pack $toolbar.sep2 -side left -fill y -pady 2 -padx 2
    pack $toolbar.mergechoicelbl -side left -padx 2
    pack $toolbar.m12_$bp $toolbar.m1_$bp $toolbar.m2_$bp $toolbar.m21_$bp \
      -side left -padx 2
    pack $toolbar.sep3 -side left -fill y -pady 2 -padx 2
    pack $toolbar.diffnavlbl -side left -pady 2 -padx 2
    pack $toolbar.first_$bp $toolbar.last_$bp $toolbar.prev_$bp \
      $toolbar.next_$bp -side left -pady 2 -padx 2
    pack $toolbar.sep4 -side left -fill y -pady 2 -padx 2
    pack $toolbar.center_$bp -side left -pady 2 -padx 1
    pack $toolbar.sep5 -side left -fill y -pady 2 -padx 2
    pack $toolbar.bkmklbl -side left -padx 2
    pack $toolbar.bkmkset_$bp $toolbar.bkmkclear_$bp -side left -pady 2 -padx 2
    pack $toolbar.sep6 -side left -fill y -pady 2 -padx 2

    foreach b [info commands $toolbar.mark*] {
        pack $b -side left -fill y -pady 2 -padx 2
    }

    foreach b [info commands $toolbar.mark*] {
        $b configure -relief $opts(relief)
    }
    foreach b [info commands $toolbar.*_$bp] {
        $b configure -relief $opts(relief)
    }

    # Radiobuttons ignore relief configuration if they have an image, so we
    # set their borderwidth to 0 if we want them flat.
    if {$opts(relief) == "flat" && $opts(toolbarIcons)} {
        set bord 0
    } else {
        set bord 1
    }
    foreach b [info commands $toolbar.m1*] {
        $b configure -bd $bord
    }
    foreach b [info commands $toolbar.m2*] {
        $b configure -bd $bord
    }
}

proc reconfigure-toolbar {} {
    debug-info "reconfigure-toolbar"
    global w

    foreach button [winfo children $w(toolbar)] {
        pack forget $button
    }

    pack-toolbuttons $w(toolbar)
}

proc build-status {} {
    debug-info "build-status"
    global w
    global g

    frame $w(status) -bd 0

    set w(statusLabel) $w(status).label
    set w(statusCurrent) $w(status).current

    # MacOS has a resize handle in the bottom right which will sit
    # on top of whatever is placed there. So, we'll add a little bit
    # of whitespace there. It's harmless, so we'll do it on all of the
    # platforms.
    label $w(status).blank -image nullImage -width 16 -bd 1 -relief sunken

    label $w(statusCurrent) -textvariable g(statusCurrent) -anchor e \
      -width 14 -borderwidth 1 -relief sunken -padx 4 -pady 2
    label $w(statusLabel) -textvariable g(statusInfo) -anchor w -width 1 \
      -borderwidth 1 -relief sunken -pady 2
    pack $w(status).blank -side right -fill y

    pack $w(statusCurrent) -side right -fill y -expand n
    pack $w(statusLabel) -side left -fill both -expand y
}

###############################################################################
# handles events over the map
###############################################################################
proc handleMapEvent {event y} {
    global opts
    global w
    global g
    #debug-info "handleMapEvent $event $y"

    switch -- $event {
    B1-Press {
            set ty1 [lindex $g(thumbBbox) 1]
            set ty2 [lindex $g(thumbBbox) 3]
            if {$y >= $ty1 && $y <= $ty2} {
                set g(mapScrolling) 1

                # this captures the negative delta between the mouse press \
                  and the top
                # of the thumbbox. It's used so when we scroll by moving the 
                # mouse, we can keep this distance constant. This is how all
                # scrollbars work, and it's what the user expects.
                set g(thumbDeltaY) [expr -1 * ($y - $ty1 - 2)]

            }
        }
    B1-Motion {
            if {[info exists g(mapScrolling)]} {
                incr y $g(thumbDeltaY)

                map-seek $y
            }
        }
    B1-Release {
            show-info ""
            set ty1 [lindex $g(thumbBbox) 1]
            set ty2 [lindex $g(thumbBbox) 3]
            # if we release over the trough (*not* over the thumb)
            # just scroll by the size of the thumb
            if {$y < $ty1 || $y > $ty2} {
                if {$y < $ty1} {
                    # if vertical scrollbar syncing is turned on,
                    # all the other windows should toe the line
                    # appropriately...
                    $w(RightText) yview scroll -1 pages
                } else {
                    $w(RightText) yview scroll 1 pages
                }

            } else {
                # do nothing
            }

            catch {unset g(mapScrolling)}
        }
    }
}

# makes a toolbar "separator"
proc toolsep {w} {
    label $w -image [image create photo] -highlightthickness 0 -bd 1 -width 0 \
      -relief groove
    return $w
}

proc toolbutton {w args} {
    global tcl_platform
    global opts
    global g

    # create the button
    eval button $w $args

    # add minimal tooltip-like support
    bind $w <Enter> [list toolbutton:handleEvent <Enter> %W]
    bind $w <Leave> [list toolbutton:handleEvent <Leave> %W]
    bind $w <FocusIn> [list toolbutton:handleEvent <FocusIn> %W]
    bind $w <FocusOut> [list toolbutton:handleEvent <FocusOut> %W]

    $w configure -relief $opts(relief)

    return $w
}

# handle events in our fancy toolbuttons...
proc toolbutton:handleEvent {event w {isToolbutton 1}} {
    global g
    global opts

    switch -- $event {
    "<Enter>" {
            showTooltip button $w
            if {$opts(fancyButtons) && $isToolbutton && [$w cget -state] == \
              "normal"} {
                $w configure -relief raised
            }
        }
    "<Leave>" {
            set g(statusInfo) ""
            if {$opts(fancyButtons) && $isToolbutton} {
                $w configure -relief flat
            }
        }
    "<FocusIn>" {
            showTooltip button $w
            if {$opts(fancyButtons) && $isToolbutton && [$w cget -state] == \
              "normal"} {
                $w configure -relief raised
            }
        }
    "<FocusOut>" {
            set g(statusInfo) ""
            if {$opts(fancyButtons) && $isToolbutton} {
                $w configure -relief flat
            }
        }
    }
}

###############################################################################
# move the map thumb to correspond to current shown merge...
###############################################################################
proc map-move-thumb {y1 y2} {
    global g
    global w

    set thumbheight [expr {($y2 - $y1) * $g(mapheight)}]
    if {$thumbheight < $g(thumbMinHeight)} {
        set thumbheight $g(thumbMinHeight)
    }

    if {![info exists g(mapwidth)]} {
        set g(mapwidth) 0
    }
    set x1 1
    set x2 [expr {$g(mapwidth) - 3}]

    # why -2? it's the thickness of our border...
    set y1 [expr {int(($y1 * $g(mapheight)) - 2)}]
    if {$y1 < 0} {
        set y1 0
    }

    set y2 [expr {$y1 + $thumbheight}]
    if {$y2 > $g(mapheight)} {
        set y2 $g(mapheight)
        set y1 [expr {$y2 - $thumbheight}]
    }

    set dx1 [expr {$x1 + 1}]
    set dx2 [expr {$x2 - 1}]
    set dy1 [expr {$y1 + 1}]
    set dy2 [expr {$y2 - 1}]

    $w(mapCanvas) coords thumbUL $x1 $y2 $x1 $y1 $x2 $y1 $dx2 $dy1 $dx1 $dy1 \
      $dx1 $dy2
    $w(mapCanvas) coords thumbLR $dx1 $y2 $x2 $y2 $x2 $dy1 $dx2 $dy1 $dx2 \
      $dy2 $dx1 $dy2

    set g(thumbBbox) [list $x1 $y1 $x2 $y2]
    set g(thumbHeight) $thumbheight
}

###############################################################################
# Bind keys for Next, Prev, Center, Merge choices 1 and 2
#
# N.B. This is GROSS! It might have been necessary in earlier versions,
# but now I think it needs a serious rewriite. We are now overriding
# the text widget, so we can probably just disable the insert and delete
# commands, and use something like insert_ and delete_ internally.
###############################################################################
proc common-navigation {args} {
    global w

    bind . <Control-f> do-find

    foreach widget $args {
        # this effectively disables the widget, without having to
        # resort to actually disabling the widget (the latter which
        # has some annoying side effects). What we really want is to
        # only disable keys that get inserted, but that's difficult
        # to do, and this works almost as well...
        bind $widget <KeyPress> {break}

        bind $widget <Alt-KeyPress> {continue}

        bind $widget <<Paste>> {break}


        # ... but now we need to restore some navigation key bindings
        # which got lost because we disable all keys. Since we are
        # attaching bindings that duplicate class bindings, we need
        # to be sure and include the break, so the events don't fire
        # twice (once for the widget, once for the class). There is
        # probably a much better way to do all this, but I'm too
        # lazy to figure it out...
        foreach event [list Next Prior Up Down Left Right Home End] {
            foreach modifier [list {} Shift Control Shift-Control] {
                set binding [bind Text <${modifier}${event}>]
                if {[string length $binding] > 0} {
                    bind $widget "<${modifier}${event}>" "
                        ${binding}
                        break
                    "
                }
            }
        }

        # these bindings allow control-f, tab and shift-tab to work
        # in spite of the fact we bound Any-KeyPress to a null action
        bind $widget <Control-f> continue

        bind $widget <Tab> continue

        bind $widget <Shift-Tab> continue


        bind $widget <c> "
            center
            break
        "
        bind $widget <n> "
            move 1
            break
        "
        bind $widget <p> "
            move -1
            break
        "
        bind $widget <f> "
            move first
            break
        "
        bind $widget <l> "
            move last
            break
        "
        bind $widget <q> "
            do-exit
            break
        "
        bind $widget <r> "
            recompute-diff
            break
        "
        bind $widget <Return> "
            moveNearest $widget mark insert
            break
        "

        # these bindings keep Alt- modified keys from triggering
        # the above actions. This way, any Alt combinations that
        # should open a menu will...
        foreach key [list c n p f l] {
            bind $widget <Alt-$key> {continue}
        }

        bind $widget <Double-1> "
            moveNearest $widget xy %x %y
            break
        "

        bind $widget <Key-1> "
            do-merge-choice 1
            break
        "
        bind $widget <Key-2> "
            do-merge-choice 2
            break
        "
        bind $widget <Key-3> "
            do-merge-choice 12
            break
        "
        bind $widget <Key-4> "
            do-merge-choice 21
            break
        "
    }
}

###############################################################################
# set or clear a "diff mark" -- a hot button to move to a particular diff
###############################################################################
proc diffmark {option {diff -1}} {
    debug-info "diffmark ($option $diff)"
    global g
    global w

    if {$diff == -1} {
        set diff $g(pos)
    }

    set widget $w(toolbar).mark$diff

    switch -- $option {
    activate {
            move $diff 0 1
        }
    mark {
            if {![winfo exists $widget]} {
                toolbutton $widget -text "\[$diff\]" -command [list diffmark \
                  activate $diff] -bd 1 -pady 1
                pack $widget -side left -padx 2
                set g(tooltip,$widget) "Diff Marker: Jump to diff record \
                  number $diff"
            }
            update-display
        }
    clear {
            if {[winfo exists $widget]} {
                destroy $widget
                catch {unset g(tooltip,$widget)}
            }
            update-display
        }
    clearall {
            set bookmarks [info commands $w(toolbar).mark*]
            if {[llength $bookmarks] > 0} {
                foreach widget $bookmarks {
                    destroy $widget
                    catch {unset g(tooltip,$widget)}
                }
            }
            update-display
        }
    }
}

###############################################################################
# Customize the display (among other things).
###############################################################################
proc customize {} {
    debug-info "customize"
    global pref
    global g
    global w
    global opts
    global tmpopts
    global tcl_platform

    catch {destroy $w(preferences)}
    toplevel $w(preferences)

    wm title $w(preferences) "$g(name) Preferences"
    wm transient $w(preferences) .
    wm group $w(preferences) .

    if {$g(windowingSystem) == "aqua"} {
        setAquaDialogStyle $w(preferences)
    }

    wm withdraw $w(preferences)

    # the button frame...
    frame $w(preferences).buttons -bd 0
    button $w(preferences).buttons.dismiss -width 8 -text "Dismiss" \
      -command {destroy $w(preferences)}
    button $w(preferences).buttons.apply -width 8 -text "Apply" \
      -command {apply 1}
    button $w(preferences).buttons.save -width 8 -text "Save" -command save

    button $w(preferences).buttons.help -width 8 -text "Help" \
      -command do-help-preferences

    pack $w(preferences).buttons -side bottom -fill x
    pack $w(preferences).buttons.dismiss -side right -padx 10 -pady 5
    pack $w(preferences).buttons.help -side right -padx 10 -pady 5
    pack $w(preferences).buttons.save -side right -padx 1 -pady 5
    pack $w(preferences).buttons.apply -side right -padx 1 -pady 5

    # a series of checkbuttons to act as a poor mans notebook tab
    frame $w(preferences).notebook -bd 0
    pack $w(preferences).notebook -side top -fill x -pady 4
    set pagelist {}

    # Radiobuttons without indicators look rather sucky on MacOSX, so
    # we'll tweak the style for that platform
    if {$::tcl_platform(os) == "Darwin"} {
        set indicatoron true
    } else {
        set indicatoron false
    }

    foreach page [list General Display Appearance] {
        set frame $w(preferences).f$page
        lappend pagelist $frame
        set rb $w(preferences).notebook.f$page
        radiobutton $rb -command "customize-selectPage $frame" \
          -variable g(prefPage) -value $frame -height 2 -text $page \
          -indicatoron $indicatoron -width 10 -borderwidth 1

        pack $rb -side left

        frame $frame -bd 2 -relief groove -width 400 -height 300
    }
    set g(prefPage) $w(preferences).fGeneral

    # make sure our labels are defined
    customize-initLabels

    # this is an option that we support internally, but don't give
    # the user a way to directly edit (right now, anyway). But we
    # need to make sure tmpopts knows about it
    set tmpopts(customCode) $opts(customCode)

    # General
    set count 0
    set frame $w(preferences).fGeneral
    foreach key {diffcmd ignoreblanksopt tmpdir editor geometry} {
        label $frame.l$count -text "$pref($key): " -anchor w
        set tmpopts($key) $opts($key)
        entry $frame.e$count -textvariable tmpopts($key) -width 50 -bd 2 \
          -relief sunken

        grid $frame.l$count -row $count -column 0 -sticky w -padx 5 -pady 2
        grid $frame.e$count -row $count -column 1 -sticky ew -padx 5 -pady 2

        incr count
    }

    # this is just for filler...
    label $frame.filler -text {}
    grid $frame.filler -row $count
    incr count

    foreach key {fancyButtons toolbarIcons autocenter syncscroll autoselect} {
        label $frame.l$count -text "$pref($key): " -anchor w
        set tmpopts($key) $opts($key)
        checkbutton $frame.c$count -indicatoron true -text "$pref($key)" \
          -justify left -onvalue 1 -offvalue 0 -variable tmpopts($key)

        set tmpopts($key) $opts($key)

        if {$key == "fancyButtons" && $g(windowingSystem) == "aqua"} {
            # Skipit - nothing to do
            incr count
            continue
        }

        grid $frame.c$count -row $count -column 0 -columnspan 2 -sticky w \
          -padx 5

        incr count
    }

    grid columnconfigure $frame 0 -weight 0
    grid columnconfigure $frame 1 -weight 1

    # this, in effect, adds a hidden row at the bottom which takes
    # up any extra room

    grid rowconfigure $frame $count -weight 1

    # pack this window for a brief moment, and compute the window
    # size. We'll do this for each "page" and find the largest
    # size to be the size of the dialog
    pack $frame -side right -fill both -expand y
    update idletasks
    set maxwidth [winfo reqwidth $w(preferences)]
    set maxheight [winfo reqheight $w(preferences)]
    pack forget $frame

    # Appearance
    set frame $w(preferences).fAppearance
    set count 0
    foreach key {textopt difftag deltag instag chgtag currtag bytetag \
      inlinetag overlaptag} {
        label $frame.l$count -text "$pref($key): " -anchor w
        set tmpopts($key) $opts($key)
        entry $frame.e$count -textvariable tmpopts($key) -bd 2 -relief sunken

        grid $frame.l$count -row $count -column 0 -sticky w -padx 5 -pady 2
        grid $frame.e$count -row $count -column 1 -sticky ew -padx 5 -pady 2

        incr count
    }
    grid columnconfigure $frame 0 -weight 0
    grid columnconfigure $frame 1 -weight 1

    # tabstops are placed after a little extra whitespace, since it is
    # slightly different than all of the other options (ie: it's not
    # a list of widget options)
    frame $frame.sep$count -bd 0 -height 4
    grid $frame.sep$count -row $count -column 0 -stick ew -columnspan 2 \
      -padx 5 -pady 2
    incr count

    set key "tabstops"
    set tmpopts($key) $opts($key)
    label $frame.l$count -text "$pref($key):" -anchor w
    set tmpopts($key) $opts($key)
    entry $frame.e$count -textvariable tmpopts($key) -bd 2 -relief sunken \
      -width 3
    grid $frame.l$count -row $count -column 0 -sticky w -padx 5 -pady 2
    grid $frame.e$count -row $count -column 1 -sticky w -padx 5 -pady 2
    incr count

    # add a tiny bit of validation, so the user can only enter numbers
    trace variable tmpopts($key) w [list validate integer]

    # this, in effect, adds a hidden row at the bottom which takes
    # up any extra room

    grid rowconfigure $frame $count -weight 1

    pack $frame -side right -fill both -expand y
    update idletasks
    set maxwidth [max $maxwidth [winfo reqwidth $w(preferences)]]
    set maxheight [max $maxheight [winfo reqheight $w(preferences)]]
    pack forget $frame

    # Display
    set frame $w(preferences).fDisplay
    set row 0

    # Option fields
    # Note that the order of the list is used to determine
    # the layout. So, if you add something to the list pay
    # attention to how it affects things.
    #
    # an x means an empty column; a - means an empty row
    set col 0
    foreach key [list showln tagln showcbs tagcbs showmap colorcbs \
      showlineview tagtext ignoreblanks showinline1 x showinline2 x] {

        if {$key == "x"} {
            set col [expr {$col ? 0 : 1}]
            if {$col == 0} {
                incr row
            }
            continue
        }

        if {$key == "-"} {
            frame $frame.f${row} -bd 0 -height 4
            grid $frame.f${row} -row $row -column 0 -columnspan 2 -padx 20 \
              -pady 4 -sticky nsew
            set col 1 ;# will force next column to zero and incr row

        } else {

            checkbutton $frame.c${row}${col} -indicatoron true \
              -text "$pref($key)" -onvalue 1 -offvalue 0 -variable tmpopts($key)

            set tmpopts($key) $opts($key)

            grid $frame.c${row}$col -row $row -column $col -sticky w -padx 5
        }

        set col [expr {$col ? 0 : 1}]
        if {$col == 0} {
            incr row
        }
    }

    grid columnconfigure $frame 0 -weight 0
    grid columnconfigure $frame 1 -weight 0
    grid columnconfigure $frame 2 -weight 0
    grid columnconfigure $frame 3 -weight 0
    grid columnconfigure $frame 4 -weight 1

    # add validation to make only one of the showinline# options are set
    trace variable tmpopts(showinline1) w [list validate-inline showinline1]
    trace variable tmpopts(showinline2) w [list validate-inline showinline2]

    # this, in effect, adds a hidden row at the bottom which takes
    # up any extra room

    grid rowconfigure $frame $row -weight 1

    pack $frame -side right -fill both -expand y
    update idletasks
    set maxwidth [max $maxwidth [winfo reqwidth $w(preferences)]]
    set maxheight [max $maxheight [winfo reqheight $w(preferences)]]
    pack forget $frame

    customize-selectPage

    # compute a reasonable location for the window...
    centerWindow $w(preferences) [list $maxwidth $maxheight]

    wm deiconify $w(preferences)
}

proc validate {type name index op} {
    global tmpopts

    # if we fail the check, attempt to do something clever
    if {![string is $type $tmpopts($index)]} {
        bell

        switch -- $type {
        integer {
                regsub -all {[^0-9]} $tmpopts($index) {} tmpopts($index)
            }
        default {
                # this should never happen. If you use this routine,
                # make sure you add cases to handle all possible
                # values of $type used by this program.
                set tmpopts($index) ""
            }
        }
    }
}

proc validate-inline {option name index op} {
    global tmpopts

    if {$tmpopts($index)} {
        if {$index == "showinline1"} {
            set tmpopts(showinline2) 0
        } elseif {$index == "showinline2"} {
            set tmpopts(showinline1) 0
        }
    }
}

proc customize-selectPage {{frame {}}} {
    global g w

    if {$frame == ""} {
        set frame $g(prefPage)
    }

    pack forget $w(preferences).fGeneral
    pack forget $w(preferences).fAppearance
    pack forget $w(preferences).fDisplay
    pack forget $w(preferences).fBehavior
    pack $frame -side right -fill both -expand y
}

###############################################################################
# define the labels for the preferences. This is done outside of
# the customize proc since the labels are used in the help text.
###############################################################################
proc customize-initLabels {} {
    global pref

    set pref(diffcmd) {diff command}
    set pref(ignoreblanksopt) {Ignore blanks option}
    set pref(ignoreblanks) {Ignore blanks when diffing}
    set pref(textopt) {Text widget options}
    set pref(bytetag) {Tag options for characters in line view}
    set pref(difftag) {Tag options for diff regions}
    set pref(currtag) {Tag options for the current diff region}
    set pref(inlinetag) {Tag options for diff region inline differences}
    set pref(deltag) {Tag options for deleted diff region}
    set pref(instag) {Tag options for inserted diff region}
    set pref(chgtag) {Tag options for changed diff region}
    set pref(overlaptag) {Tag options for overlap diff region}
    set pref(geometry) {Text window size}
    set pref(tmpdir) {Directory for scratch files}
    set pref(editor) {Program for editing files}

    set pref(fancyButtons) {Windows-style toolbar buttons}
    set pref(showlineview) {Show current line comparison window}
    set pref(showinline1) {Show inline diffs (byte comparisons)}
    set pref(showinline2) {Show inline diffs (recursive matching algorithm)}
    set pref(showmap) {Show graphical map of diffs}
    set pref(showln) {Show line numbers}
    set pref(showcbs) {Show change bars}
    set pref(autocenter) {Automatically center current diff region}
    set pref(syncscroll) {Synchronize scrollbars}
    set pref(toolbarIcons) {Use icons instead of labels in the toolbar}

    set pref(colorcbs) {Color change bars to match the diff map}
    set pref(tagtext) {Highlight file contents}
    set pref(tagcbs) {Highlight change bars}
    set pref(tagln) {Highlight line numbers}
    set pref(tabstops) {Tab stops}

    set pref(autoselect) "Automatically select the nearest diff region while \
      scrolling"
}

###############################################################################
# Apply customization changes.
###############################################################################
proc apply {{remark 0}} {
    debug-info "apply ($remark)"
    global opts
    global tmpopts
    global w
    global pref
    global screenWidth
    global screenHeight
    global tk_version

    grid propagate $w(client) t
    if {! [file isdirectory $tmpopts(tmpdir)]} {
        do-error "Invalid temporary directory $tmpopts(tmpdir)"
    }

    if {[catch "
        $w(LeftText) configure $tmpopts(textopt)
        $w(RightText) configure $tmpopts(textopt)
        $w(BottomText) configure $tmpopts(textopt)
    "]} {
        do-error "Invalid text widget setting: \n\n'$tmpopts(textopt)'"
        eval "$w(LeftText)   configure $opts(textopt)"
        eval "$w(RightText)  configure $opts(textopt)"
        eval "$w(BottomText) configure $opts(textopt)"
        return
    }

    # the text options must be ok. Configure the other text widgets
    # similarly
    eval "$w(LeftCB)    configure $tmpopts(textopt)"
    eval "$w(LeftInfo)  configure $tmpopts(textopt)"
    eval "$w(RightCB)   configure $tmpopts(textopt)"
    eval "$w(RightInfo) configure $tmpopts(textopt)"

    set gridsize [wm grid .]
    set gridx [lindex $gridsize 2]
    set gridy [lindex $gridsize 3]
    debug-info " wm grid is $gridx x $gridy"

    set maxunitsx [expr {$screenWidth / $gridx}]
    set maxunitsy [expr {$screenHeight / $gridy}]
    debug-info "   max X is $maxunitsx units"
    debug-info "   max Y is $maxunitsy units"
    set halfmax [expr {$maxunitsx / 2}]

    if {$tmpopts(geometry) == "" || [catch {scan $tmpopts(geometry) \
      "%dx%d" width height} result]} {
        do-error "invalid geometry setting: $tmpopts(geometry)"
        return
    }
    debug-info " width $width  halfmax $halfmax"
    set maxw [expr {$halfmax - 18}]
    debug-info " maxw $maxw"
    if {$width > $maxw} {
        set width $maxw
    }
    # re-center map
    if {$tk_version < 8.4} {
      grid columnconfigure $w(client) 0 -weight 1
      grid columnconfigure $w(client) 2 -weight 1
    } else {
      grid columnconfigure $w(client) 0 -weight 100 -uniform a
      grid columnconfigure $w(client) 2 -weight 100 -uniform a
    }

    if {[catch {$w(LeftText) configure -width $width -height $height} result]} {
        do-error "invalid geometry setting: $tmpopts(geometry)"
        return
    }
    $w(RightText) configure -width $width -height $height

    $w(LeftLabel) configure -width $width
    $w(RightLabel) configure -width $width

    grid forget $w(LeftLabel)
    grid forget $w(RightLabel)
    grid $w(LeftLabel) -row 0 -column 0 -sticky ew
    grid $w(RightLabel) -row 0 -column 2 -sticky ew

    foreach tag {difftag currtag inlinetag deltag instag chgtag overlaptag} {
        foreach win [list $w(LeftText) $w(LeftInfo) $w(LeftCB) $w(RightText) \
          $w(RightInfo) $w(RightCB)] {
            if {[catch "$win tag configure $tag $tmpopts($tag)"]} {
                do-error "Invalid settings for \"$pref($tag)\": \
                \n\n'$tmpopts($tag)' is not a valid option string"
                eval "$win tag configure $tag $opts($tag)"
                return
            }
        }
    }

    if {[catch "$w(BottomText) tag configure diff $tmpopts(bytetag)"]} {
        do-error "Invalid settings for \"$pref(bytetag)\": \
        \n\n'$tmpopts(bytetag)' is not a valid option string"
        eval "$w(BottomText) tag configure diff $opts(bytetag)"
        return
    }

    # tabstops require a little extra work. We need to figure out
    # the width of an "m" in the widget's font, then multiply that
    # by the tab stop width. For the bottom text widget the first tabstop
    # is adjusted by two to take into consideration the fact that we
    # add two bytes to each line (ie: "< " or "> ").
    set cwidth [font measure [$w(LeftText) cget -font] "m"]
    set tabstops [expr {$cwidth * $tmpopts(tabstops)}]
    $w(LeftText) configure -tabs $tabstops
    $w(RightText) configure -tabs $tabstops

    $w(BottomText) configure -tabs [list [expr {$tabstops +($cwidth * 2)}] \
      [expr {2*$tabstops +($cwidth * 2)}]]

    if {[info exists w(mergeText)] && [winfo exists $w(mergeText)]} {
        $w(mergeText) configure -tabs $tabstops
    }

    # set opts to the values from tmpopts
    foreach key {autocenter autoselect chgtag colorcbs currtag deltag diffcmd \
      difftag inlinetag editor fancyButtons geometry ignoreblanks \
      ignoreblanksopt instag overlaptag showcbs showlineview showln showmap \
      syncscroll tabstops tagcbs tagln tagtext textopt tmpdir toolbarIcons} {
        set opts($key) $tmpopts($key)
    }
    if {$opts(fancyButtons)} {
        set opts(relief) flat
    } else {
        set opts(relief) raised
    }

    # determine if we need to redo the inline diffs to avoid needless rediff
    if {$opts(showinline1) != $tmpopts(showinline1) || $opts(showinline2) != \
      $tmpopts(showinline2)} {
        set opts(showinline1) $tmpopts(showinline1)
        set opts(showinline2) $tmpopts(showinline2)
        recompute-diff
    }

    # reconfigure the toolbar buttons
    reconfigure-toolbar

    # remark all the diff regions, show (or hide) the line numbers,
    # change bars and diff map, and we are done
    if {$remark} {
        remark-diffs
    }

    do-show-linenumbers
    do-show-changebars
    do-show-map
    do-show-lineview
    do-show-ignoreblanks
    grid propagate $w(client) f
}

###############################################################################
# Save customization changes.
###############################################################################
proc save {} {
    debug-info "save"
    global g
    global tmpopts rcfile tcl_platform
    global pref

    if {[file exists $rcfile]} {
        file rename -force $rcfile "$rcfile~"
    }

    set fid [open $rcfile w]

    # put the tkdiff version in the file. It might be handy later
    puts $fid "# This file was generated by $g(name) $g(version)"
    puts $fid "# [clock format [clock seconds]]\n"
    puts $fid "set prefsFileVersion {$g(version)}\n"

    # now, put all of the preferences in the file
    foreach key [lsort [array names pref]] {
        regsub "\n" $pref($key) "\n# " comment
        puts $fid "# $comment"
        puts $fid "define $key {$tmpopts($key)}\n"
    }

    # ... and any custom code
    puts $fid "# custom code"
    puts $fid "# put any custom code you want to be executed in the"
    puts $fid "# following block. This code will be automatically executed"
    puts $fid "# after the GUI has been set up but before the diff is "
    puts $fid "# performed. Use this code to customize the interface if"
    puts $fid "# you so desire."
    puts $fid "#  "
    puts $fid "# Even though you can't (as of version 3.09) edit this "
    puts $fid "# code via the preferences dialog, it will be automatically"
    puts $fid "# saved and restored if you do a SAVE from that dialog."
    puts $fid ""
    puts $fid "# Unless you really know what you are doing, it is probably"
    puts $fid "# wise to leave this unmodified."
    puts $fid ""
    puts $fid "define customCode {\n[string trim $tmpopts(customCode) \n]\n}\n"

    close $fid

    if {$::tcl_platform(platform) == "windows"} {
        file attribute $rcfile -hidden 1
    }
}

###############################################################################
# Text has scrolled, update scrollbars and synchronize windows
###############################################################################
proc hscroll-sync {id args} {
    global g opts
    global w

    # If ignore_event is true, we've already taken care of scrolling.
    # We're only interested in the first event.
    if {$g(ignore_hevent,$id)} {
        return
    }

    # Scrollbar sizes
    set size1 [expr {[lindex [$w(LeftText) xview] 1] - [lindex \
      [$w(LeftText) xview] 0]}]
    set size2 [expr {[lindex [$w(RightText) xview] 1] - [lindex \
      [$w(RightText) xview] 0]}]

    if {$opts(syncscroll) || $id == 1} {
        set start [lindex $args 0]

        if {$id != 1} {
            set start [expr {$start * $size2 / $size1}]
        }
        $w(LeftHSB) set $start [expr {$start + $size1}]
        $w(LeftText) xview moveto $start
        set g(ignore_hevent,1) 1
    }
    if {$opts(syncscroll) || $id == 2} {
        set start [lindex $args 0]
        if {$id != 2} {
            set start [expr {$start * $size1 / $size2}]
        }
        $w(RightHSB) set $start [expr {$start + $size2}]
        $w(RightText) xview moveto $start
        set g(ignore_hevent,2) 1
    }

    # This forces all the event handlers for the view alterations
    # above to trigger, and we lock out the recursive (redundant)
    # events using ignore_event.
    update idletasks

    # Restore to normal
    set g(ignore_hevent,1) 0
    set g(ignore_hevent,2) 0
}

###############################################################################
# Text has scrolled, update scrollbars and synchronize windows
###############################################################################
proc vscroll-sync {windowlist id y0 y1} {
    global g opts
    global w

    if {$id == 1} {
        $w(LeftVSB) set $y0 $y1
    } else {
        $w(RightVSB) set $y0 $y1
    }

    # if syncing is disabled, we're done. This prevents a nasty
    # set of recursive calls
    if {[info exists g(disableSyncing)]} {
        return
    }

    # set the flag; this makes sure we only get called once
    set g(disableSyncing) 1

    # scroll the other windows on the same side as this window
    foreach window $windowlist {
        $window yview moveto $y0
    }

    eval map-move-thumb $y0 $y1

    # Select nearest visible diff region, if the appropriate
    # options are set
    if {$opts(syncscroll) && $opts(autoselect) && $g(count) > 0} {
        set winhalf [expr {[winfo height $w(RightText)] / 2}]
        set result [find-diff [expr {int([$w(RightText) index @1,$winhalf])}]]
        set i [lindex $result 0]

        # have we found a diff other than the current diff?
        if {$i != $g(pos)} {
            # Also, make sure the diff is visible. If not, we won't
            # change the current diff region...
            set topline [$w(RightText) index @0,0]
            set bottomline [$w(RightText) index @0,10000]
            foreach {line s1 e1 s2 e2 type} $g(scrdiff,$i) { }
            if {$s1 >= $topline && $s1 <= $bottomline} {
                move $i 0 1
            }
        }
    }

    # if syncing is turned on, scroll other windows.
    # Annoyingly, sometimes the *Text windows won't scroll properly,
    # at least under windows. And I can't for the life of me figure
    # out why. Maybe a bug in tk?
    if {$opts(syncscroll)} {
        if {$id == 1} {

            $w(RightText) yview moveto $y0
            $w(RightInfo) yview moveto $y0
            $w(RightCB) yview moveto $y0
            $w(RightVSB) set $y0 $y1

        } else {

            $w(LeftText) yview moveto $y0
            $w(LeftInfo) yview moveto $y0
            $w(LeftCB) yview moveto $y0
            $w(LeftVSB) set $y0 $y1
        }
    }

    # we apparently automatically process idle events after this
    # proc is called. Once that is done we'll unset our flag
    after idle {catch {unset g(disableSyncing)}}
}

###############################################################################
# Make a miniature map of the diff regions
###############################################################################
proc create-map {name mapwidth mapheight} {
    global g
    global w
    global map
    global opts

    set map $name

    # Text widget always contains blank line at the end
    set lines [expr {double([$w(LeftText) index end]) - 2}]
    set factor [expr {$mapheight / $lines}]

    # We add some transparent stuff to make the map fill the canvas
    # in order to receive mouse events at the very bottom.
    $map blank
    $map put \#000 -to 0 $mapheight $mapwidth $mapheight

    # Line numbers start at 1, not at 0.
    for {set i 1} {$i <= $g(count)} {incr i} {
        #         scan $g(scrdiff,$i) "%s %d %d %d %d %s" line s1 e1 s2 e2 type
        foreach {line s1 e1 s2 e2 type} $g(scrdiff,$i) { }

        set y [expr {int(($s2 - 1) * $factor) + $g(mapborder)}]
        set size [expr {round(($e2 - $s2 + 1) * $factor)}]
        if {$size < 1} {
            set size 1
        }
        switch -- $type {
        "d" {
                set color $opts(mapdel)
            }
        "a" {
                set color $opts(mapins)
            }
        "c" {
                set color $opts(mapchg)
            }
        }
        if {[info exists g(overlap$i)]} {
            set color yellow
        }

        $map put $color -to 0 $y $mapwidth [expr {$y + $size}]

    }

    # let's draw a rectangle to simulate a scrollbar thumb. The size
    # isn't important since it will get resized when map-move-thumb
    # is called...
    $w(mapCanvas) create line 0 0 0 0 -width 1 -tags thumbUL -fill white
    $w(mapCanvas) create line 1 1 1 1 -width 1 -tags thumbLR -fill black
    $w(mapCanvas) raise thumb

    # now, move the thumb
    eval map-move-thumb [$w(LeftText) yview]

}

###############################################################################
# Resize map to fit window size
###############################################################################
proc map-resize {args} {
    global g opts
    global w

    set mapwidth [winfo width $w(map)]
    set g(mapborder) [expr {[$w(map) cget -borderwidth] + [$w(map) cget \
      -highlightthickness]}]
    set mapheight [expr {[winfo height $w(map)] - $g(mapborder) * 2}]

    # We'll get a couple of "resize" events, so don't draw a map
    # unless we've got the diffs and the map size has changed
    if {$g(count) == 0 || $mapheight == $g(mapheight)} {
        return
    }

    # If we don't have a map and don't want one, don't make one
    if {$g(mapheight) == 0 && $opts(showmap) == 0} {
        return
    }

    # This seems to happen on Windows!? _After_ the map is drawn the first time
    # another event triggers and [winfo height $w(map)] is then 0...
    if {$mapheight < 1} {
        return
    }

    set g(mapheight) $mapheight
    set g(mapwidth) $mapwidth
    create-map map $mapwidth $mapheight
}

###############################################################################
# scroll to diff region nearest to y
###############################################################################
proc map-scroll {y} {
    global g
    global w
    global opts

    set yview [expr {double($y) / double($g(mapheight))}]
    # Show text corresponding to map
    catch {$w(RightText) yview moveto $yview} result
    update idletasks

    # Select the diff region closest to the middle of the screen
    set winhalf [expr {[winfo height $w(RightText)] / 2}]
    set result [find-diff [expr {int([$w(RightText) index @1,$winhalf])}]]
    move [lindex $result 0] 0 0

    if {$opts(autocenter)} {
        center
    }

    if {$g(showmerge)} {
        merge-center
    }
}

###############################################################################
# Toggle showing the line comparison window
###############################################################################
proc do-show-lineview {{showLineview {}}} {
    global opts
    global w

    if {$showLineview != {}} {
        set opts(showlineview) $showLineview
    }

    if {$opts(showlineview)} {
        grid $w(BottomText) -row 3 -column 0 -sticky ew -columnspan 4
    } else {
        grid forget $w(BottomText)
    }
}

###############################################################################
# Toggle showing inline comparison
###############################################################################
proc do-show-inline1 {{showInline1 {}}} {
    global opts

    if {$showInline1 != {}} {
        puts "passed in value=$showInline1"
        set opts(showinline1) $showInline1
    }

    # mutually disjoint options
    if {$opts(showinline1)} {
        set opts(showinline2) 0
    }
    recompute-diff
}

proc do-show-inline2 {{showInline2 {}}} {
    global opts

    if {$showInline2 != {}} {
        set opts(showinline2) $showInline2
    }

    # mutually disjoint options
    if {$opts(showinline2)} {
        set opts(showinline1) 0
    }
    recompute-diff
}

###############################################################################
# Toggle showing map or not
###############################################################################
proc do-show-map {{showMap {}}} {
    global opts
    global w

    if {$showMap != {}} {
        set opts(showmap) $showMap
    }

    if {$opts(showmap)} {
        grid $w(map) -row 1 -column 1 -stick ns
    } else {
        grid forget $w(map)
    }
}

###############################################################################
# Find the diff nearest to $line.
# Returns "$i $newtop" where $i is the index of the diff region
# and $newtop is the new top line in the window to the right.
###############################################################################
proc find-diff {line} {
    global g
    global w

    set top $line
    set newtop [expr {$top - int([$w(LeftText) index end]) + \
      int([$w(RightText) index end])}]

    for {set low 1; set high $g(count); set i [expr {($low + $high) / 2}]} \
      {$i >= $low} {set i [expr {($low + $high) / 2}]} {

        foreach {line s1 e1 s2 e2 type} $g(scrdiff,$i) { }

        if {$s1 > $top} {
            set newtop [expr {$top - $s1 + $s2}]
            set high [expr {$i-1}]
        } else {
            set low [expr {$i+1}]
        }
    }

    # do some range checking...
    set i [max 1 [min $i $g(count)]]

    # If next diff is closer than the one found, use it instead
    if {$i > 0 && $i < $g(count)} {
        set nexts1 [lindex $g(scrdiff,[expr {$i + 1}]) 1]
        set e1 [lindex $g(scrdiff,$i) 2]
        if {$nexts1 - $top < $top - $e1} {
            incr i
        }
    }

    return [list $i $newtop]
}

###############################################################################
# Calculate number of lines in diff region
# pos            Diff number
# version   1 or 2, left or right window version
# screen    1 for screen size, 0 for original diff size
###############################################################################
proc diff-size {pos version {screen 0}} {
    global g

    if {$screen} {
        set diff scrdiff
    } else {
        set diff pdiff
    }

    foreach {thisdiff s(1) e(1) s(2) e(2) type} $g($diff,$pos) { }

    switch -- $version {
    1 {
            set lines [expr {$e(1) - $s(1) + 1}]
            if {$type == "a"} {
                incr lines -1
            }
        }
    2 {
            set lines [expr {$e(2) - $s(2) + 1}]
            if {$type == "d"} {
                incr lines -1
            }
        }
    12 -
    21 {
            set lines [expr {$e(1) - $s(1) + $e(2) - $s(2) + 1}]
        }
    }
    return $lines
}

###############################################################################
# Toggle showing merge preview or not
###############################################################################
proc do-show-merge {{showMerge ""}} {
    debug-info "do-show-merge ($showMerge)"
    global g
    global w

    if {$showMerge != ""} {
        set g(showmerge) $showMerge
    }

    if {$g(showmerge)} {
        watch-cursor
        if {! [info exists w(mergeText]} {
            merge-read-file
            merge-add-marks
        }
        wm deiconify .merge
        $w(mergeText) configure -state disabled
        focus -force $w(mergeText)
        merge-center
    } else {
        wm withdraw $w(merge)
    }
    debug-info "  ...restore-cursor from do-show-merge"
    restore-cursor
}

###############################################################################
# Create Merge preview window
###############################################################################
proc merge-create-window {} {
    debug-info "merge-create-window"
    global opts
    global w
    global g

    set top .merge
    set w(merge) $top

    catch {destroy $top}

    toplevel $top
    set rx [winfo rootx .]
    set ry [winfo rooty .]
    set px [winfo width .]
    set py [winfo height .]
    #debug-info "  rx $rx  ry $ry  px $px  py $py"
    set x [expr {$rx + $px / 4}]
    set y [expr {$ry + $py / 2}]
    wm geometry $top "+${x}+$y"

    wm group $top .
    wm title $top "$g(name) Merge Preview"

    frame $top.frame -bd 1 -relief sunken
    pack $top.frame -side top -fill both -expand y -padx 10 -pady 10

    set w(mergeText) $top.frame.text
    set w(mergeVSB) $top.frame.vsb
    set w(mergeHSB) $top.frame.hsb
    set w(mergeDismiss) $top.dismiss
    set w(mergeWrite) $top.mergeWrite
    set w(mergeWriteAndExit) $top.mergeWriteAndExit
    set w(mergeExit) $top.mergeExit
    set w(mergeRecenter) $top.mergeRecenter

    # Window and scrollbars
    scrollbar $w(mergeHSB) -orient horizontal -command [list $w(mergeText) \
      xview]
    scrollbar $w(mergeVSB) -orient vertical -command [list $w(mergeText) yview]

    text $w(mergeText) -bd 0 -takefocus 1 -yscrollcommand [list $w(mergeVSB) \
      set] -xscrollcommand [list $w(mergeHSB) set]

    grid $w(mergeText) -row 0 -column 0 -sticky nsew
    grid $w(mergeVSB) -row 0 -column 1 -sticky ns
    grid $w(mergeHSB) -row 1 -column 0 -sticky ew

    grid rowconfigure $top.frame 0 -weight 1
    grid rowconfigure $top.frame 1 -weight 0

    grid columnconfigure $top.frame 0 -weight 1
    grid columnconfigure $top.frame 1 -weight 0

    # buttons
    button $w(mergeRecenter) -width 8 -text "ReCenter" -underline 0 \
      -command merge-center

    button $w(mergeDismiss) -width 8 -text "Dismiss" -underline 0 \
      -command "do-show-merge 0"

    if {$g(mergefileset)} {
        button $w(mergeWrite) -width 8 -text "Save" -underline 0 \
          -command [list popup-merge merge-write-file]
        button $w(mergeWriteAndExit) -width 8 -text "Save & Exit" \
          -underline 8 -command {
            popup-merge merge-write-file
            exit
        }
    } else {
        button $w(mergeWrite) -width 8 -text "Save..." -underline 0 \
          -command [list popup-merge merge-write-file]
        button $w(mergeWriteAndExit) -width 10 -text "Save & Exit..." \
          -underline 8 -command {
            popup-merge merge-write-file
            exit
        }
    }
    button $w(mergeExit) -width 8 -text "Exit $g(name)" -underline 0 \
      -command {exit}

    pack $w(mergeDismiss) -side right -pady 5 -padx 10
    pack $w(mergeRecenter) -side right -pady 5 -padx 1
    pack $w(mergeWrite) -side right -pady 5 -padx 1
    pack $w(mergeWriteAndExit) -side right -pady 5 -padx 1
    pack $w(mergeExit) -side right -pady 5 -padx 1

    eval $w(mergeText) configure $opts(textopt)
    foreach tag {difftag currtag} {
        eval $w(mergeText) tag configure $tag $opts($tag)
    }

    # adjust the tabstops
    set cwidth [font measure [$w(mergeText) cget -font] "m"]
    set tabstops [expr {$cwidth * $opts(tabstops)}]
    $w(mergeText) configure -tabs $tabstops

    wm protocol $w(merge) WM_DELETE_WINDOW {do-show-merge 0}

    # adjust the tag priorities a bit...
    $w(mergeText) tag raise sel
    $w(mergeText) tag raise currtag difftag

    common-navigation $w(mergeText)

    if {! $g(showmerge)} {
        wm withdraw $w(merge)
    }
}

###############################################################################
# Read original file (Left window file) into merge preview window.
# Not so good if it has changed.
###############################################################################
proc merge-read-file {} {
    debug-info "merge-read-file"
    global finfo
    global w

    # hack; need to find a cleaner way...
    catch {destroy .merge}
    merge-create-window

    set hndl [open "$finfo(pth,1)" r]
    $w(mergeText) configure -state normal
    $w(mergeText) delete 1.0 end
    $w(mergeText) insert 1.0 [read $hndl]
    close $hndl

    # If last line doesn't end with a newline, add one. Important when
    # writing out the merge preview.
    if {![regexp {\.0$} [$w(mergeText) index "end-1lines lineend"]]} {
        $w(mergeText) insert end "\n"
    }
    $w(mergeText) configure -state disabled
}

###############################################################################
# Write merge preview to file
###############################################################################
proc merge-write-file {} {
    global g
    global w

    set hndl [open "$g(mergefile)" w]
    set text [$w(mergeText) get 1.0 end-1lines]
    puts -nonewline $hndl $text
    close $hndl
}

###############################################################################
# Add a mark where each diff begins and tag diff regions so they are visible.
# Assumes text is initially the bare original (Left) version.
###############################################################################
proc merge-add-marks {} {
    global g
    global w

    # mark all lines first, so selection won't mess up line numbers
    for {set i 1} {$i <= $g(count)} {incr i} {
        foreach [list thisdiff s1 e1 s2 e2 type] $g(pdiff,$i) { }
        #        set delta [expr {$type == "a" ? 1 : 0}]
        #        $w(mergeText) mark set mark$i $s1.0+${delta}lines
        if {$type == "a"} {
            incr s1
        }
        $w(mergeText) mark set mark$i $s1.0
        $w(mergeText) mark gravity mark$i left
    }

    # if a 3-way merge, select right window as needed
    if {$g(ancfileset) && $g(count) > 0} {
        #
        # If there was something different between file1
        # and the ancestor, pick the left window, but...
        #
        for {set i 1} {$i <= $g(count)} {incr i} {
            set s1 [lindex $g(pdiff,$i) 1]
            set s2 [lindex $g(pdiff,$i) 2]
            for {set p $s1} {$p <= $s2} {incr p} {
                if {[info exists g(diff3l$p)]} {
                    set g(merge$i) 1
                    break
                }
            }
        }

        #
        # ... if there was a diff between file2 and the ancestor,
        # then file2 takes precedence
        #
        for {set i 1} {$i <= $g(count)} {incr i} {
            set s1 [lindex $g(pdiff,$i) 3]
            set s2 [lindex $g(pdiff,$i) 4]
            for {set p $s1} {$p <= $s2} {incr p} {
                if {[info exists g(diff3r$p)]} {
                    set g(merge$i) 2
                    break
                }
            }
        }
    }

    # select merged lines
    for {set i 1} {$i <= $g(count)} {incr i} {
        foreach [list thisdiff s1 e1 s2 e2 type] $g(pdiff,$i) { }

        if {$g(merge$i) == 1} {
            # (If it's an insert it's not visible)
            if {$type != "a"} {
                set lines [expr {$e1 - $s1 + 1}]
                $w(mergeText) tag add difftag mark$i mark$i+${lines}lines
            }
        } else {
            # Insert right window version
            merge-select-version $i 1 2
        }
    }

    # Tag current
    if {$g(count) > 0} {
        set pos $g(pos)
        set lines [diff-size $pos $g(merge$pos)]
        $w(mergeText) tag add currtag mark$pos "mark$pos+${lines}lines"
    }
}

###############################################################################
# Add a mark where each diff begins
# pos               diff index
# oldversion   1 or 2, previous merge choice
# newversion   1 or 2, new merge choice
###############################################################################
proc merge-select-version {pos oldversion newversion} {
    global g
    global w

    catch {
        switch -- $oldversion {
        1 -
        2 {set oldlines [diff-size $pos $oldversion]}
        12 -
        21 {set oldlines [expr {[diff-size $pos 1] + [diff-size $pos 2]}]}
        }
        $w(mergeText) configure -state normal
        $w(mergeText) delete mark$pos "mark${pos}+${oldlines}lines"
        $w(mergeText) configure -state disabled
    }

    # Screen coordinates
    foreach {thisdiff s(1) e(1) s(2) e(2) type} $g(scrdiff,$pos) { }

    # Get the text directly from window
    switch -- $newversion {
    1 {
            set newlines [diff-size $pos 1]
            set newtext [$w(LeftText) get $s(1).0 $s(1).0+${newlines}lines]
        }
    2 {
            set newlines [diff-size $pos 2]
            set newtext [$w(RightText) get $s(2).0 $s(2).0+${newlines}lines]
        }
    12 {
            set newlines [diff-size $pos 1]
            set newtext [$w(LeftText) get $s(1).0 $s(1).0+${newlines}lines]
            set newlines [diff-size $pos 2]
            append newtext [$w(RightText) get $s(2).0 $s(2).0+${newlines}lines]
            incr newlines [diff-size $pos 1]
        }
    21 {
            set newlines [diff-size $pos 2]
            set newtext [$w(RightText) get $s(2).0 $s(2).0+${newlines}lines]
            set newlines [diff-size $pos 1]
            append newtext [$w(LeftText) get $s(1).0 $s(1).0+${newlines}lines]
            incr newlines [diff-size $pos 2]
        }
    }

    # Insert it
    $w(mergeText) configure -state normal
    $w(mergeText) insert mark$pos $newtext diff
    $w(mergeText) configure -state disabled
    if {$pos == $g(pos)} {
        $w(mergeText) tag add currtag mark$pos "mark${pos}+${newlines}lines"
    }
}

###############################################################################
# Center the merge region in the merge window
###############################################################################
proc merge-center {} {
    global g
    global w

    # bail if there are no diffs
    if {$g(count) == 0} {
        return
    }
    # Size of diff in lines of text
    set difflines [diff-size $g(pos) $g(merge$g(pos))]
    set yview [$w(mergeText) yview]
    # Window height in percent
    set ywindow [expr {[lindex $yview 1] - [lindex $yview 0]}]
    # First line of diff
    set firstline [$w(mergeText) index mark$g(pos)]
    # Total number of lines in window
    set totallines [$w(mergeText) index end]

    if {$difflines / $totallines < $ywindow} {
        # Diff fits in window, center it
        $w(mergeText) yview moveto [expr {($firstline + $difflines / 2) / \
          $totallines - $ywindow / 2}]
    } else {
        # Diff too big, show top part
        $w(mergeText) yview moveto [expr {($firstline - 1) / $totallines}]
    }
}

###############################################################################
# Update the merge preview window with the current merge choice
# newversion   1 or 2, new merge choice
###############################################################################
proc do-merge-choice {newversion} {
    debug-info "do-merge-choice ($newversion)"
    global g opts
    global w

    if {! [info exists w(mergeText)] || ! [winfo exists $w(mergeText)]} {
       return
    }
    $w(mergeText) configure -state normal
    merge-select-version $g(pos) $g(merge$g(pos)) $newversion
    $w(mergeText) configure -state disabled

    set g(merge$g(pos)) $newversion
    if {$g(showmerge) && $opts(autocenter)} {
        merge-center
    }
    set g(toggle) $newversion
}

###############################################################################
# Extract the start and end lines for file1 and file2 from the diff
# stored in "line".
###############################################################################
proc extract {line} {
    # the line darn well better be of the form <range><op><range>,
    # where op is one of "a","c" or "d". range will either be a
    # single number or two numbers separated by a comma.

    # is this a cool regular expression, or what? :-)
    regexp {([0-9]*)(,([0-9]*))?([a-z])([0-9]*)(,([0-9]*))?} $line matchvar \
      s1 x e1 op s2 x e2
    if {[string length $e1] == 0} {
        set e1 $s1
    }
    if {[string length $e2] == 0} {
        set e2 $s2
    }

    if {[info exists s1] && [info exists s2]} {
        #         return "$line $s1 $e1 $s2 $e2 $op"
        return [list $line $s1 $e1 $s2 $e2 $op]
    } else {
        fatal-error "Cannot parse output from diff:\n$line"
    }

}

###############################################################################
# Insert blank lines to match added/deleted lines in other file
###############################################################################
proc add-lines {pos} {
    global g
    global w
    global opts

    # Figure out which lines we need to address...
    foreach [list thisdiff s1 e1 s2 e2 type] $g(pdiff,$pos) { }

    set size(1) [expr {$e1 - $s1}]
    set size(2) [expr {$e2 - $s2}]

    incr s1 $g(delta,1)
    incr s2 $g(delta,2)

    # Figure out what kind of diff we're dealing with
    switch -- $type {
    "a" {
            set lefttext " " ;# insert
            set righttext "+"
            set idx 1
            set count [expr {$size(2) + 1}]

            incr s1
            incr size(2)
        }
    "d" {
            set lefttext "-" ;# delete
            set righttext " "
            set idx 2
            set count [expr {$size(1) + 1}]

            incr s2
            incr size(1)
        }
    "c" {
            set lefttext "!" ;# change
            set righttext "!" ;# change
            if {$g(ancfileset)} {
                set change $g(pdiff,$g(count))
                set leftBegin [lindex $change 1]
                set leftEnd [lindex $change 2]
                set rightBegin [lindex $change 3]
                set rightEnd [lindex $change 4]

                set changeLeft 0
                set changeRight 0
                for {set i $leftBegin} {$i <= $leftEnd} {incr i} {
                    if {[info exists g(diff3l$i)]} {
                        set changeLeft 1
                        break
                    }
                }
                if {$changeLeft} {
                    for {set i $rightBegin} {$i <= $rightEnd} {incr i} {
                        if {[info exists g(diff3r$i)]} {
                            set changeRight 1
                            break
                        }
                    }
                }
                if {$changeLeft && $changeRight} {
                    set lefttext "?" ;# overlap
                    set righttext "?" ;# overlap
                    set g(overlap$pos) 1
                }
            }
            set idx [expr {$size(1) < $size(2) ? 1 : 2}]
            set count [expr {abs($size(1) - $size(2))}]

            incr size(1)
            incr size(2)
        }
    }

    # Put plus signs in left info column
    if {$idx == 1} {
        set textWidget $w(LeftText)
        set infoWidget $w(LeftInfo)
        set cbWidget $w(LeftCB)
        #        set blank "++++++\n"
        set blank "      \n"
    } else {
        set textWidget $w(RightText)
        set infoWidget $w(RightInfo)
        set cbWidget $w(RightCB)
        set blank "      \n"
    }

    # Insert blank lines to match other window
    set line [expr {$s1 + $size($idx)}]
    for {set i 0} {$i < $count} {incr i} {
        $textWidget insert $line.0 "\n"
        $infoWidget insert $line.0 $blank
        $cbWidget insert $line.0 "\n"
    }

    incr size($idx) $count
    set e1 [expr {$s1 + $size(1) - 1}]
    set e2 [expr {$s2 + $size(2) - 1}]
    incr g(delta,$idx) $count

    # Insert change bars or text to show what has changed.
    $w(RightCB) configure -state normal
    $w(LeftCB) configure -state normal
    for {set i $s1} {$i <= $e1} {incr i} {
        $w(LeftCB) insert $i.0 $lefttext
        $w(RightCB) insert $i.0 $righttext
    }

    # Save the diff block in window coordinates
    set g(scrdiff,$g(count)) [list $thisdiff $s1 $e1 $s2 $e2 $type]

    set g(scrinline,$pos) 0
    if {$opts(showinline1) || $opts(showinline2)} {
        if {$type == "c"} {
            set numlines [max [expr {$e1-$s1+1}] [expr {$e2-$s2+1}]]
            for {set i 0} {$i < $numlines} {incr i} {
                set l1 [expr $s1+$i]
                set l2 [expr $s2+$i]
                if {$opts(showinline1)} {
                    find-inline-diff-byte $pos $l1 $l2 [$w(LeftText) get \
                      $l1.0 $l1.end] [$w(RightText) get $l2.0 $l2.end]
                } else {
                    find-inline-diff-ratcliff $pos $l1 $l2 [$w(LeftText) get \
                      $l1.0 $l1.end] [$w(RightText) get $l2.0 $l2.end]
                }
            }
        }
    }
}

###############################################################################
# Add a tag to a region.
###############################################################################
proc add-tag {wgt tag start end type new {exact 0}} {
    global g

    $wgt tag add $tag $start.0 [expr {$end + 1}].0

}

proc add-inline-tag {wgt tag line startcol endcol} {
    $wgt tag add $tag $line.$startcol $line.$endcol
}

###############################################################################
# Change the tag for a diff region.
# 'pos' is the index in the diff array
# If 'oldtag' is present, first remove it from the region
# If 'setpos' is non-zero, make sure the region is visible.
# Returns the diff expression.
###############################################################################
proc set-tag {pos newtag {oldtag ""} {setpos 0}} {
    global g opts
    global w

    # Figure out which lines we need to address...
    if {![info exists g(scrdiff,$pos)]} {
        return
    }
    foreach {thisdiff s1 e1 s2 e2 dt} $g(scrdiff,$pos) { }

    # Remove old tag
    if {"$oldtag" != ""} {
        set e1next "[expr {$e1 + 1}].0"
        set e2next "[expr {$e2 + 1}].0"
        $w(LeftText) tag remove $oldtag $s1.0 $e1next
        $w(LeftInfo) tag remove $oldtag $s1.0 $e1next
        $w(RightText) tag remove $oldtag $s2.0 $e2next
        $w(RightInfo) tag remove $oldtag $s2.0 $e2next
        $w(LeftCB) tag remove $oldtag $s1.0 $e1next
        $w(RightCB) tag remove $oldtag $s2.0 $e2next
        catch {
            set lines [diff-size $pos $g(merge$pos)]
            $w(mergeText) tag remove $oldtag mark$pos "mark${pos}+${lines}lines"
        }
    }

    switch -- $dt {
    "d" {
            set coltag deltag
            set rcbtag " "
            set lcbtag "-"
        }
    "a" {
            set coltag instag
            set rcbtag "+"
            set lcbtag " "
        }
    "c" {
            set coltag chgtag
            set rcbtag "!"
            set lcbtag "!"
        }
    }
    if {[info exists g(overlap$pos)]} {
        set coltag overlaptag
        set rcbtag "?"
        set lcbtag "?"
    }
    # Add new tag
    if {$opts(tagtext)} {
        add-tag $w(LeftText) $newtag $s1 $e1 $dt 1
        add-tag $w(RightText) $newtag $s2 $e2 $dt 1
        add-tag $w(RightText) $coltag $s2 $e2 $dt 1
    }
    if {$opts(tagcbs)} {
        if {$opts(colorcbs)} {
            add-tag $w(LeftCB) $lcbtag $s1 $e1 $dt 1
            add-tag $w(RightCB) $rcbtag $s2 $e2 $dt 1
        } else {
            add-tag $w(LeftCB) $newtag $s1 $e1 $dt 1
            add-tag $w(RightCB) $newtag $s2 $e2 $dt 1
            add-tag $w(RightCB) $coltag $s2 $e2 $dt 1
        }

    }
    if {$opts(tagln)} {
        add-tag $w(LeftInfo) $newtag $s1 $e1 $dt 1
        add-tag $w(RightInfo) $newtag $s2 $e2 $dt 1
        add-tag $w(RightInfo) $coltag $s2 $e2 $dt 1
    }

    catch {
        set lines [diff-size $pos $g(merge$pos)]
        $w(mergeText) tag add $newtag mark$pos "mark${pos}+${lines}lines"
    }

    # Move the view on both text widgets so that the new region is
    # visible.
    if {$setpos} {
        if {$opts(autocenter)} {
            center
        } else {
            $w(LeftText) see $s1.0
            $w(RightText) see $s2.0
            $w(LeftText) mark set insert $s1.0
            $w(RightText) mark set insert $s2.0

            if {$g(showmerge)} {
                $w(mergeText) see mark$pos
            }
        }
    }

    # make sure the sel tag has the highest priority
    foreach window [list LeftText RightText LeftCB RightCB LeftInfo RightInfo] {
        $w($window) tag raise sel
        $w($window) tag raise inlinetag
    }

    return $thisdiff
}

###############################################################################
# moves to the diff nearest the insertion cursor or the mouse click,
# depending on $mode (which should be either "xy" or "mark")
###############################################################################
proc moveNearest {window mode args} {
    switch -- $mode {
    "xy" {
            set x [lindex $args 0]
            set y [lindex $args 1]
            set index [$window index @$x,$y]

            set line [expr {int($index)}]
            set diff [find-diff $line]
        }
    "mark" {
            set index [$window index [lindex $args 0]]
            set line [expr {int($index)}]
            set diff [find-diff $line]
        }
    }

    # ok, we have an index
    move [lindex $diff 0] 0 1
}

###############################################################################
###############################################################################
proc moveTo {window value} {
    global w
    global g
    # we know that the value is prefixed by the nunber/index of
    # the diff the user wants. So, just grab that out of the string
    regexp {([0-9]+) *:} $value matchVar index
    move $index 0 1
}

###############################################################################
# this is called when the user scrolls the map thumb interactively.
###############################################################################
proc map-seek {y} {
    global g
    global w

    set yview [expr {(double($y) / double($g(mapheight)))}]

    # Show text corresponding to map;
    $w(RightText) yview moveto $yview
}

###############################################################################
# Move the "current" diff indicator (i.e. go to the next or previous diff
# region if "relative" is 1; go to an absolute diff number if "relative"
# is 0).
###############################################################################
proc move {value {relative 1} {setpos 1}} {
    #debug-info "move $value $relative $setpos"
    global g
    global w

    if {$value == "first"} {
        set value 1
        set relative 0
    }
    if {$value == "last"} {
        set value $g(count)
        set relative 0
    }

    # Remove old 'curr' tag
    set-tag $g(pos) difftag currtag

    # Bump 'pos' (one way or the other).
    if {$relative} {
        set g(pos) [expr {$g(pos) + $value}]
    } else {
        set g(pos) $value
    }

    # Range check 'pos'.
    set g(pos) [max $g(pos) 1]
    set g(pos) [min $g(pos) $g(count)]

    # Set new 'curr' tag
    set g(currdiff) [set-tag $g(pos) currtag "" $setpos]

    # update the buttons..
    #debug-info "   ...update-display from move"
    update-display

}

proc update-display {} {
    #debug-info "update-display"
    global g
    global w

    #debug-info "  init_OK $g(initOK)"
    #debug-info "  started $g(started)"
    #if {!$g(started)} return
    if {!$g(initOK)} {
        # disable darn near everything

        foreach b [list rediff find prevDiff firstDiff nextDiff lastDiff \
          centerDiffs mergeChoice1 mergeChoice2 mergeChoice12 mergeChoice21] {
            $w(${b}_im) configure -state disabled
            $w(${b}_tx) configure -state disabled
        }
        foreach menu [list $w(popupMenu) $w(viewMenu)] {
            $menu entryconfigure "Previous*" -state disabled
            $menu entryconfigure "First*" -state disabled
            $menu entryconfigure "Next*" -state disabled
            $menu entryconfigure "Last*" -state disabled
            $menu entryconfigure "Center*" -state disabled
        }
        $w(popupMenu) entryconfigure "Find..." -state disabled
        $w(popupMenu) entryconfigure "Find Nearest*" -state disabled
        $w(popupMenu) entryconfigure "Edit*" -state disabled

        $w(editMenu) entryconfigure "Find*" -state disabled
        $w(editMenu) entryconfigure "Edit File 1" -state disabled
        $w(editMenu) entryconfigure "Edit File 2" -state disabled

        $w(fileMenu) entryconfigure "Write*" -state disabled
        $w(fileMenu) entryconfigure "Recompute*" -state disabled

        $w(mergeMenu) entryconfigure "Show*" -state disabled
        $w(mergeMenu) entryconfigure "Write*" -state disabled

        $w(markMenu) entryconfigure "Mark*" -state disabled
        $w(markMenu) entryconfigure "Clear*" -state disabled

    } else {
        # these are always enabled, assuming we have properly
        # diffed a couple of files
        $w(popupMenu) entryconfigure "Find..." -state normal
        $w(popupMenu) entryconfigure "Find Nearest*" -state normal
        $w(popupMenu) entryconfigure "Edit*" -state normal

        foreach b [list rediff find prevDiff firstDiff nextDiff lastDiff \
          centerDiffs] {
            $w(${b}_im) configure -state normal
            $w(${b}_tx) configure -state normal
        }
        foreach b [list mergeChoice1 mergeChoice2 mergeChoice12 mergeChoice21] {
            $w(${b}_im) configure -state normal
            $w(${b}_tx) configure -state normal
        }

        $w(editMenu) entryconfigure "Find*" -state normal
        $w(editMenu) entryconfigure "Edit File 1" -state normal
        $w(editMenu) entryconfigure "Edit File 2" -state normal

        $w(fileMenu) entryconfigure "Write*" -state normal
        $w(fileMenu) entryconfigure "Recompute*" -state normal

        $w(mergeMenu) entryconfigure "Show*" -state normal
        $w(mergeMenu) entryconfigure "Write*" -state normal

        $w(find_im) configure -state normal
        $w(find_tx) configure -state normal

        # Hmmm.... on my Mac the combobox flashes if we don't add this
        # check. Is this a bug in AquaTk, or in my combobox... :-|
        if {[$w(combo) cget -state] != "normal"} {
            $w(combo) configure -state normal
        }
    }

    # Update the toggles.
    if {$g(count)} {
        set g(toggle) $g(merge$g(pos))
    }

    # update the status line
    set g(statusCurrent) "$g(pos) of $g(count)"
    show-info $g(statusCurrent)

    # update the combobox. We don't want its command to fire, so
    # we'll disable it temporarily
    $w(combo) configure -commandstate "disabled"
    set i [expr {$g(pos) - 1}]
    $w(combo) configure -value [lindex [$w(combo) list get 0 end] $i]
    $w(combo) selection clear
    $w(combo) configure -commandstate "normal"

    # update the widgets
    if {$g(pos) <= 1} {
        foreach buttonpref {im tx} {
            $w(prevDiff_$buttonpref) configure -state disabled
            $w(firstDiff_$buttonpref) configure -state disabled
        }
        $w(popupMenu) entryconfigure "Previous*" -state disabled
        $w(popupMenu) entryconfigure "First*" -state disabled
        $w(viewMenu) entryconfigure "Previous*" -state disabled
        $w(viewMenu) entryconfigure "First*" -state disabled
    } else {
        foreach buttonpref {im tx} {
            $w(prevDiff_$buttonpref) configure -state normal
            $w(firstDiff_$buttonpref) configure -state normal
        }
        $w(popupMenu) entryconfigure "Previous*" -state normal
        $w(popupMenu) entryconfigure "First*" -state normal
        $w(viewMenu) entryconfigure "Previous*" -state normal
        $w(viewMenu) entryconfigure "First*" -state normal
    }

    if {$g(pos) >= $g(count)} {
        foreach buttonpref {im tx} {
            $w(nextDiff_$buttonpref) configure -state disabled
            $w(lastDiff_$buttonpref) configure -state disabled
        }
        $w(popupMenu) entryconfigure "Next*" -state disabled
        $w(popupMenu) entryconfigure "Last*" -state disabled
        $w(viewMenu) entryconfigure "Next*" -state disabled
        $w(viewMenu) entryconfigure "Last*" -state disabled
    } else {
        foreach buttonpref {im tx} {
            $w(nextDiff_$buttonpref) configure -state normal
            $w(lastDiff_$buttonpref) configure -state normal
        }
        $w(popupMenu) entryconfigure "Next*" -state normal
        $w(popupMenu) entryconfigure "Last*" -state normal
        $w(viewMenu) entryconfigure "Next*" -state normal
        $w(viewMenu) entryconfigure "Last*" -state normal
    }

    if {$g(count) > 0} {
        $w(popupMenu) entryconfigure "Center*" -state normal
        $w(viewMenu) entryconfigure "Center*" -state normal
        $w(markMenu) entryconfigure "Mark*" -state normal

        foreach buttonpref {im tx} {
            $w(centerDiffs_$buttonpref) configure -state normal
            $w(mergeChoice1_$buttonpref) configure -state normal
            $w(mergeChoice2_$buttonpref) configure -state normal
            $w(mergeChoice12_$buttonpref) configure -state normal
            $w(mergeChoice21_$buttonpref) configure -state normal
        }
        catch { $w(mergeChoiceLabel) configure -state normal }

    } else {
        foreach buttonpref {im tx} {
            $w(centerDiffs_$buttonpref) configure -state disabled
            $w(mergeChoice1_$buttonpref) configure -state disabled
            $w(mergeChoice2_$buttonpref) configure -state disabled
            $w(mergeChoice12_$buttonpref) configure -state disabled
            $w(mergeChoice21_$buttonpref) configure -state disabled
        }
        catch { $w(mergeChoiceLabel) configure -state disabled }
        $w(popupMenu) entryconfigure "Center*" -state disabled
        $w(viewMenu) entryconfigure "Center*" -state disabled

        $w(markMenu) entryconfigure "Mark*" -state disabled
    }

    # the mark clear button should only be enabled if there is
    # presently a mark at the current diff record
    set widget $w(toolbar).mark$g(pos)
    if {[winfo exists $widget]} {
        $w(markMenu) entryconfigure "Clear*" -state normal
        $w(markMenu) entryconfigure "Mark*" -state disabled
        foreach buttonpref {im tx} {
            $w(markClear_$buttonpref) configure -state normal
            $w(markSet_$buttonpref) configure -state disabled
        }
    } else {
        $w(markMenu) entryconfigure "Clear*" -state disabled
        $w(markMenu) entryconfigure "Mark*" -state normal
        foreach buttonpref {im tx} {
            $w(markClear_$buttonpref) configure -state disabled
            $w(markSet_$buttonpref) configure -state normal
        }
    }
}

###############################################################################
# Center the top line of the CDR in each window.
###############################################################################
proc center {} {
    global g
    global w

    if {! [info exists g(scrdiff,$g(pos))]} {return}
    #scan $g(scrdiff,$g(pos)) "%s %d %d %d %d %s" dummy s1 e1 s2 e2 dt
    foreach {dummy s1 e1 s2 e2 dt} $g(scrdiff,$g(pos)) { }

    # Window requested height in pixels
    set opix [winfo reqheight $w(LeftText)]
    # Window requested lines
    set olin [$w(LeftText) cget -height]
    # Current window height in pixels
    set npix [winfo height $w(LeftText)]

    # Visible lines
    set winlines [expr {$npix * $olin / $opix}]
    # Lines in diff
    set diffsize [max [expr {$e1 - $s1 + 1}] [expr {$e2 - $s2 + 1}]]

    if {$diffsize < $winlines} {
        set h [expr {($winlines - $diffsize) / 2}]
    } else {
        set h 2
    }

    set o [expr {$s1 - $h}]
    if {$o < 0} {
        set o 0
    }
    set n [expr {$s2 - $h}]
    if {$n < 0} {
        set n 0
    }

    $w(LeftText) mark set insert $s1.0
    $w(RightText) mark set insert $s2.0
    $w(LeftText) yview $o
    $w(RightText) yview $n

    if {$g(showmerge)} {
        merge-center
    }
}

###############################################################################
# Change the state on all of the diff-sensitive buttons.
###############################################################################
proc buttons {{newstate "normal"}} {
    global w
    $w(combo) configure -state $newstate
    foreach buttonpref {im tx} {
        $w(prevDiff_$buttonpref) configure -state $newstate
        $w(nextDiff_$buttonpref) configure -state $newstate
        $w(firstDiff_$buttonpref) configure -state $newstate
        $w(lastDiff_$buttonpref) configure -state $newstate
        $w(centerDiffs_$buttonpref) configure -state $newstate
    }
}

###############################################################################
# Wipe the slate clean...
###############################################################################
proc wipe {} {
    debug-info "wipe"
    global g

    set g(pos) 0
    set g(count) 0
    set g(diff) ""
    set g(currdiff) ""

    set g(delta,1) 0
    set g(delta,2) 0
}

###############################################################################
# Wipe all data and all windows
###############################################################################
proc wipe-window {} {
    debug-info "wipe-window"
    global g
    global w

    wipe

    foreach mod {Left Right} {
        $w(${mod}Text) configure -state normal
        $w(${mod}Text) tag remove difftag 1.0 end
        $w(${mod}Text) tag remove currtag 1.0 end
        $w(${mod}Text) tag remove inlinetag 1.0 end
        $w(${mod}Text) delete 1.0 end

        $w(${mod}Info) configure -state normal
        $w(${mod}Info) delete 1.0 end
        $w(${mod}CB) configure -state normal
        $w(${mod}CB) delete 1.0 end
    }

    catch {
        $w(mergeText) configure -state normal
        $w(mergeText) delete 1.0 end
        eval $w(mergeText) tag delete [$w(mergeText) tag names]
        $w(mergeText) configure -state disabled
    }

    if {[string length $g(destroy)] > 0} {
        eval $g(destroy)
        set g(destroy) ""
    }

    $w(combo) list delete 0 end
    buttons disabled

    diffmark clearall
}

###############################################################################
# Mark difference regions and build up the combobox
###############################################################################
proc mark-diffs {} {
    debug-info "mark-diffs"
    global g
    global w

    set numdiff [llength "$g(diff)"]

    set g(count) 0


    # ain't this clever? We want to update the display as soon as
    # we've marked enough diffs to fill the display so the user will
    # have the impression we're fast. But, we don't want this
    # want this code to slow us down too much, so we'll put the
    # code in a variable and delete it when its no longer needed.
    set hack {
        # for now, just pick a number out of thin air. Ideally
        # we'd compute the number of lines that are visible and
        # use that, but I'm too lazy today...
        if {$g(count) > 25} {
            update idletasks
            set hack {}
        }
    }

    foreach d $g(diff) {
        set result [extract $d]
        if {$result != ""} {
            incr g(count)
            set g(merge$g(count)) 1

            set g(pdiff,$g(count)) "$result"
            add-lines $g(count)

            $w(combo) list insert end [format "%-6d: %s" $g(count) $d]

            eval $hack
        }

    }

    remark-diffs
    return $g(count)
}

###############################################################################
# start a new diff from the popup dialog
###############################################################################
proc do-new-diff {} {
    global g
    global finfo

    debug-info "do-new-diff"

    set g(mergefileset) 0
    set g(mergefile) ""
    set finfo(pth,1) ""
    set finfo(pth,2) ""
    set finfo(tmp,1) 0
    set finfo(tmp,2) 0

    #foreach inf [lsort [array names finfo]] { debug-info "    $inf: \
      $finfo($inf)" }
    # Pop up the dialog to collect the args
    newDiffDialog

    # Put them together into a command
    if {[assemble-args] != 0} return

    foreach inf [lsort [array names finfo]] {
        debug-info "    $inf: $finfo($inf)"
    }

    set g(disableSyncing) 1 ;# turn off syncing until things settle down

    # remove all evidence of previous diff
    #wipe-window
    #update idletasks

    watch-cursor
    # do the diff
    do-diff

    debug-info "   move first 1 1  from do-new-diff"
    move first 1 1

    #debug-info "    ...restore-cursor from do-new-diff"
    restore-cursor

    #debug-info "    ...update-display from do-new-diff"
    update-display
    catch {unset g(disableSyncing)}
}

###############################################################################
# Remark difference regions...
###############################################################################
proc remark-diffs {} {
    debug-info "remark-diffs"
    global g
    global w
    global opts
    global pref

    # delete all known tags.
    foreach window [list $w(LeftText) $w(LeftInfo) $w(LeftCB) $w(RightText) \
      $w(RightInfo) $w(RightCB)] {
        eval $window tag delete [$window tag names]
    }
    if {[winfo exists .merge]} {
        eval $window tag delete [$w(mergeText) tag names]
    }

    # reconfigure all the tags based on the current options
    # first, the common tags:
    foreach tag {difftag currtag inlinetag deltag instag chgtag overlaptag} {
        foreach win [list $w(LeftText) $w(LeftInfo) $w(LeftCB) $w(RightText) \
          $w(RightInfo) $w(RightCB)] {
            if {[catch "$win tag configure $tag $opts($tag)"]} {
                do-error "Invalid settings for \"$pref($tag)\": \
                \n\n'$opts($tag)' is not a valid option string."
                eval "$win tag configure $tag $opts($tag)"
                return
            }
        }
    }

    # next, changebar-specific tags
    foreach widget [list $w(LeftCB) $w(RightCB)] {
        eval $widget tag configure + $opts(+)
        eval $widget tag configure - $opts(-)
        eval $widget tag configure ! $opts(!)
        eval $widget tag configure ? $opts(?)
    }

    # ... and the merge text window
    if {[winfo exists .merge]} {
        foreach tag {difftag currtag} {
            eval $w(mergeText) tag configure $tag $opts($tag)
        }
    }

    # now, reapply the tags to all the diff regions
    for {set i 1} {$i <= $g(count)} {incr i} {
        set-tag $i difftag
        # add the inline annotation
        for {set j 0} {$j < $g(scrinline,$i)} {incr j} {
            foreach {side line startcol endcol} $g(scrinline,$i,$j) { }
            if {$side == "l"} {
                add-inline-tag $w(LeftText) inlinetag $line $startcol $endcol
            } else {
                add-inline-tag $w(RightText) inlinetag $line $startcol $endcol
            }
        }
    }

    # finally, reset the current diff
    set-tag $g(pos) currtag "" 0
}


###############################################################################
# Put up some informational text.
###############################################################################
proc show-info {message} {
    global g

    set g(statusInfo) $message
    debug-info "show-info: $message"
    update idletasks
}


###############################################################################
# Trace output, enabled by a global variable
###############################################################################
proc debug-info {message} {
    global g

    if {$g(debug)} {
        puts "$message"
    }
}

###############################################################################
# Compute differences (start over, basically).
###############################################################################
proc rediff {} {
    debug-info "\nrediff"
    global g
    global opts
    global finfo
    global w

    buttons disabled

    # Read the files into their respective widgets & add line numbers.
    foreach mod {1 2} {
        if {$mod == 1} {
            set text $w(LeftText)
        } else {
            set text $w(RightText)
        }
        show-info "reading $finfo(pth,$mod)..."
        if {[catch {set hndl [open "$finfo(pth,$mod)" r]}]} {
            fatal-error "Failed to open file: $finfo(pth,$mod)"
        }
        $text insert 1.0 [read $hndl]
        close $hndl
    }

    # Diff the two files and store the summary lines into 'g(diff)'.
    if {$opts(ignoreblanks) == 1} {
        set diffcmd "$opts(diffcmd) $opts(ignoreblanksopt)  {$finfo(pth,1)} \
          {$finfo(pth,2)}"
    } else {
        set diffcmd "$opts(diffcmd) {$finfo(pth,1)} {$finfo(pth,2)}"
    }
    show-info "Executing \"$diffcmd\""

    set result [run-command "exec $diffcmd"]
    set stdout [lindex $result 0]
    set stderr [lindex $result 1]
    set exitcode [lindex $result 2]
    set g(returnValue) $exitcode

    # The exit code is 0 if there are no differences and 1 if there
    # are differences. Any other exit code means trouble
    if {$exitcode < 0 || $exitcode > 1 || $stderr != ""} {
        do-error "diff failed:\n$stderr"
    }

    set g(diff) {}
    set lines [split $stdout "\n"]

    # If there is no output and we got this far the files are equal,
    # otherwise check if the first line begins with a line number. If
    # not there was trouble and we abort. For instance, using a binary
    # file results in the message "Binary files ..." etc on stdout,
    # exit code 1. The message may wary depending on locale.
    if {$lines != "" && [string match {[0-9]*} $lines] != 1} {
        fatal-error "diff failed:\n$stdout"
    }

    # Collect all lines containing line numbers
    foreach line $lines {
        if {[string match {[0-9]*} $line]} {
            lappend g(diff) $line
        }
    }

    if {$g(ancfileset)} {

        # 3-way merge - compare 1st file (left: diff3l) with ancestor
        if {$opts(ignoreblanks) == 1} {
            set diffcmd "$opts(diffcmd) $opts(ignoreblanksopt) \
              {$finfo(pth,1)} {$g(ancfile)}"
        } else {
            set diffcmd "$opts(diffcmd) {$finfo(pth,1)} {$g(ancfile)}"
        }
        show-info "Executing \"$diffcmd\""
        set result [run-command "exec $diffcmd"]
        set stdout [lindex $result 0]
        set stderr [lindex $result 1]
        set exitcode [lindex $result 2]
        if {$exitcode < 0 || $exitcode > 1 || $stderr != ""} {
            fatal-error "diff3 failed:\n$stderr"
        }
        set lines [split $stdout "\n"]
        set g(diff3l) {}
        foreach line $lines {
            if {[string match {[0-9]*} $line]} {
                if {[regexp {^[0-9]*,} $line match]} {
                    regexp {([0-9]*),([0-9]*).*} $line matchvar s1 s2
                } else {
                    regexp {([0-9]*).*} $line matchvar s1
                    set s2 $s1
                }

                lappend g(diff3l) $s1
                for {set i $s1} {$i <= $s2} {incr i} {
                    set g(diff3l$i) 1
                }
            }
        }

        # 3-way merge - compare 2nd file (right: diff3r) with ancestor
        if {$opts(ignoreblanks) == 1} {
            set diffcmd "$opts(diffcmd) $opts(ignoreblanksopt) \
              {$finfo(pth,2)} {$g(ancfile)}"
        } else {
            set diffcmd "$opts(diffcmd) {$finfo(pth,2)} {$g(ancfile)}"
        }
        show-info "Executing \"$diffcmd\""
        set result [run-command "exec $diffcmd"]
        set stdout [lindex $result 0]
        set stderr [lindex $result 1]
        set exitcode [lindex $result 2]
        if {$exitcode < 0 || $exitcode > 1 || $stderr != ""} {
            fatal-error "diff3 failed:\n$stderr"
        }
        set lines [split $stdout "\n"]
        set g(diff3r) {}
        foreach line $lines {
            if {[string match {[0-9]*} $line]} {
                if {[regexp {^[0-9]*,} $line match]} {
                    regexp {([0-9]*),([0-9]*).*} $line matchvar s1 s2
                } else {
                    regexp {([0-9]*).*} $line matchvar s1
                    set s2 $s1
                }

                lappend g(diff3r) $s1
                for {set i $s1} {$i <= $s2} {incr i} {
                    set g(diff3r$i) 1
                }
            }
        }
    }

    # Mark up the two text widgets and go to the first diff (if there is one).
    draw-line-numbers

    show-info "Marking differences..."

    $w(LeftInfo) configure -state normal
    $w(RightInfo) configure -state normal
    $w(LeftCB) configure -state normal
    $w(RightCB) configure -state normal

    if {[mark-diffs]} {
        set g(pos) 1
        move 1 0 1
        buttons normal
    } else {
        after idle {show-info "Files are identical"}
        buttons disabled
    }

    # Prevent tampering in the line number widgets. The text
    # widgets are already taken care of
    $w(LeftInfo) configure -state disabled
    $w(RightInfo) configure -state disabled
    $w(LeftCB) configure -state disabled
    $w(RightCB) configure -state disabled
}

###############################################################################
# Set the X cursor to "watch" for a window and all of its descendants.
###############################################################################
proc watch-cursor {args} {
    debug-info "-> watch-cursor ($args)"
    global current
    global w

    . configure -cursor watch
    $w(LeftText) configure -cursor watch
    $w(RightText) configure -cursor watch
    $w(combo) configure -cursor watch
    update idletasks
}

###############################################################################
# Restore the X cursor for a window and all of its descendants.
###############################################################################
proc restore-cursor {args} {
    debug-info "-> restore-cursor ($args)"
    global current
    global w

    . configure -cursor {}
    $w(LeftText) configure -cursor {}
    $w(RightText) configure -cursor {}
    $w(combo) configure -cursor {}
    show-info ""
    update idletasks
}

###############################################################################
# Check if error was thrown by us or unexpected
###############################################################################
proc check-error {result output} {
    global g errorInfo

    if {$result && $output != "Fatal"} {
        error $result $errorInfo
    }
}


###############################################################################
# redo the current diff. Attempt to return to the same diff region,
# numerically speaking.
###############################################################################
proc recompute-diff {} {

    debug-info "recompute-diff"
    global g
    set current $g(pos)
    debug-info "current position $g(pos)"

    do-diff
    move $current 0 1
    center
}


###############################################################################
# Flash the "rediff" button and then kick off a rediff.
###############################################################################
proc do-diff {} {
    debug-info "do-diff"
    global g finfo map errorInfo
    global opts

    wipe-window
    update idletasks
    set result [catch {
        if {$g(mapheight)} {
            ## FIXME this could better a catch
            catch {$map blank}
        }

        #assemble-args
        rediff
        merge-read-file
        merge-add-marks

        # If a map exists, recreate it
        if {$opts(showmap)} {
            set g(mapheight) -1
            map-resize
        }

    } output]

    #debug-info "  result: $result   outptut: $output"
    check-error $result $output

    if {$g(mergefileset)} {
        do-show-merge 1
    }
}

###############################################################################
# Get things going...
###############################################################################
proc main {} {
    debug-info "main"
    global w
    global g errorInfo
    global startupError
    global opts
    global waitvar

    set cmd_empty [commandline]
    debug-info "  main: commandline returned $cmd_empty"
    if {! $cmd_empty} {
        assemble-args
    } else {
        newDiffDialog
        # If they cancel the dialog before doing any diffs, exit
        if {[assemble-args] != 0} {
          if {! [winfo exists .client]} {
             do-exit
          }
          # If the full UI is drawn, don't exit
          return
        }
        set cmd_empty 0
    }

    wm withdraw .
    wm protocol . WM_DELETE_WINDOW do-exit
    wm title . "$g(name) $g(version)"

    if {![catch {set windowingsystem [tk windowingsystem]}]} {
        if {$windowingsystem == "x11"} {
            # All this nonsense is necessary to use an icon bitmap that's
            # not in a separate file.
            toplevel .icw
            if {[string first "color" [winfo visual .]] >= 0} {
                label .icw.l -image deltaGif
            } else {
                label .icw.l -image delta48
            }

            pack .icw.l
            bind .icw <Button-1> "wm deiconify ."
            wm iconwindow . .icw
        }
    }

    set g(started) 1
    wipe

    if {$g(windowingSystem) == "x11"} {
        get_cde_params
    }
    if {$g(windowingSystem) == "aqua"} {
        get_aqua_params
    }

    create-display

    update

    do-show-linenumbers
    do-show-map

    # evaluate any custom code the user has
    if {[info exists opts(customCode)]} {
        if {[catch [list uplevel \#0 $opts(customCode)] error]} {
            set startupError "Error in custom code: \n\n$error"
        } else {
            update
        }
    }

    if {$cmd_empty} {
        do-new-diff
    }
    move first 1 1

    # this forces all of the various scrolling windows (line numbers,
    # change bars, etc) to get in sync.
    set yview [$w(RightText) yview]
    vscroll-sync [list $w(RightInfo) $w(LeftInfo)] 2 [lindex $yview 0] \
      [lindex $yview 1]

    wm deiconify .
    update idletasks

    if {[info exists startupError]} {
        tk_messageBox -icon warning -type ok -title "$g(name) - Error in \
          Startup File" -message $startupError
    }
}

###############################################################################
# Erase tmp files (if necessary) and destroy the application.
###############################################################################
proc del-tmp {} {
    global g

    foreach f $g(tempfiles) {
        file delete $f
    }
}

###############################################################################
# Put up a window with formatted text
###############################################################################
proc do-text-info {w title text} {
    global g

    catch "destroy $w"
    toplevel $w

    wm group $w .
    wm transient $w .
    wm title $w "$g(name) Help - $title"

    if {$g(windowingSystem) == "aqua"} {
        setAquaDialogStyle $w
    }

    set width 64
    set height 32

    frame $w.f -bd 2 -relief sunken
    pack $w.f -side top -fill both -expand y

    text $w.f.title -highlightthickness 0 -bd 0 -height 2 -wrap word \
      -width 50 -background white -foreground black

    text $w.f.text -wrap word -setgrid true -padx 20 -highlightthickness 0 \
      -bd 0 -width $width -height $height -yscroll [list $w.f.vsb set] \
      -background white -foreground black
    scrollbar $w.f.vsb -borderwidth 1 -command [list $w.f.text yview] \
      -orient vertical

    pack $w.f.vsb -side right -fill y -expand n
    pack $w.f.title -side top -fill x -expand n
    pack $w.f.text -side left -fill both -expand y

    focus $w.f.text

    button $w.done -text Dismiss -command "destroy $w"
    pack $w.done -side right -fill none -pady 5 -padx 5

    put-text $w.f.title "<ttl>$title</ttl>"
    put-text $w.f.text $text
    $w.f.text configure -state disabled

    wm geometry $w ${width}x${height}
    update idletasks
    raise $w
}

###############################################################################
# centers window w over parent
###############################################################################
proc centerWindow {w {size {}}} {
    update
    set parent .

    if {[llength $size] > 0} {
        set wWidth [lindex $size 0]
        set wHeight [lindex $size 1]
    } else {
        set wWidth [winfo reqwidth $w]
        set wHeight [winfo reqheight $w]
    }

    set pWidth [winfo reqwidth $parent]
    set pHeight [winfo reqheight $parent]
    set pX [winfo rootx $parent]
    set pY [winfo rooty $parent]

    set centerX [expr {$pX +($pWidth / 2)}]
    set centerY [expr {$pY +($pHeight / 2)}]

    set x [expr {$centerX -($wWidth / 2)}]
    set y [expr {$centerY -($wHeight / 2)}]

    if {[llength $size] > 0} {
        wm geometry $w "=${wWidth}x${wHeight}+${x}+${y}"
    } else {
        wm geometry $w "=+${x}+${y}"
    }
    update
}

###############################################################################
# The "New Diff" dialog
# In order to be able to enter only one filename if it's a revision-controlled
# file, the dialog now collects the arguments and sends them through the
# command line parser.
###############################################################################
proc newDiffDialog {} {
    debug-info "newDiffDialog"
    global g w
    global finfo

    set g(mergefile) ""
    set g(mergefileset) 0

    set waitvar {}
    set w(newDiffPopup) .newDiffPopup

    if {[winfo exists $w(newDiffPopup)]} {
        debug-info " $w(newDiffPopup) already exists, just centering"
    } else {
        debug-info " creating $w(newDiffPopup)"
        toplevel $w(newDiffPopup)

        wm group $w(newDiffPopup) .
        # Won't start as the first window on Windows if it's transient
        if {[winfo exists .client]} {
            wm transient $w(newDiffPopup) .
        }
        wm title $w(newDiffPopup) "New Diff"

        if {$g(windowingSystem) == "aqua"} {
            setAquaDialogStyle $w(newDiffPopup)
        }

        wm protocol $w(newDiffPopup) WM_DELETE_WINDOW {wm withdraw \
            $w(newDiffPopup)}
        wm withdraw $w(newDiffPopup)

        set simple [frame $w(newDiffPopup).simple -borderwidth 2 -relief groove]

        label $simple.l1 -text "File 1:"
        label $simple.l2 -text "File 2:"
        entry $simple.e1 -textvariable finfo(f,1)
        entry $simple.e2 -textvariable finfo(f,2)

        label $simple.lr1 -text "-r"
        label $simple.lr2 -text "-r"
        entry $simple.er1 -textvariable finfo(revs,1)
        entry $simple.er2 -textvariable finfo(revs,2)

        set w(newDiffPopup,entry1) $simple.e1
        set w(newDiffPopup,entry2) $simple.e2

        # we want these buttons to be the same height as
        # the entry, so we'll try to force the issue...
        button $simple.b1 -borderwidth 1 -highlightthickness 0 \
          -text "Browse..." -command [list newDiffBrowse "File 1" $simple.e1]
        button $simple.b2 -borderwidth 1 -highlightthickness 0 \
          -text "Browse..." -command [list newDiffBrowse "File 2" $simple.e2]


        # we'll use the grid geometry manager to get things lined up right...
        grid $simple.l1 -row 0 -column 0 -sticky e
        grid $simple.e1 -row 0 -column 1 -columnspan 4 -sticky nsew -pady 4
        grid $simple.b1 -row 0 -column 5 -sticky nsew -padx 4 -pady 4

        grid $simple.lr1 -row 1 -column 1
        grid $simple.er1 -row 1 -column 2
        grid $simple.lr2 -row 1 -column 3
        grid $simple.er2 -row 1 -column 4

        grid $simple.l2 -row 2 -column 0 -sticky e
        grid $simple.e2 -row 2 -column 1 -columnspan 4 -sticky nsew -pady 4
        grid $simple.b2 -row 2 -column 5 -sticky nsew -padx 4 -pady 4

        grid columnconfigure $simple 0 -weight 0

        set options [frame $w(newDiffPopup).options -borderwidth 2 \
          -relief groove]

        button $options.more -text "More" -command open-more-options

        label $options.ml -text "Merge Output"
        entry $options.me -textvariable g(mergefile)
        label $options.al -text "Ancestor"
        entry $options.ae -textvariable g(ancfile)
        label $options.l1l -text "Label for File 1"
        entry $options.l1e -textvariable finfo(userlbl,1)
        label $options.l2l -text "Label for File 2"
        entry $options.l2e -textvariable finfo(userlbl,2)

        grid $options.more -column 0 -row 0 -sticky nw
        grid columnconfigure $options -0 -weight 0

        # here are the buttons for this dialog...
        set commands [frame $w(newDiffPopup).buttons]

        button $commands.ok -text "Ok" -width 5 -default active -command {
            if {$g(mergefile) == ""} {
                set g(mergefileset) 0
            } else {
                set g(mergefileset) 1
            }
            if {$g(ancfile) == ""} {
                set g(ancfileset) 0
            } else {
                set g(ancfileset) 1
            }
            set waitvar 1
        }
        button $commands.cancel -text "Cancel" -width 5 -default normal \
          -command {
            wm withdraw $w(newDiffPopup); set waitvar 0
        }

        pack $commands.ok $commands.cancel -side left -fill none -expand y \
          -pady 4

        catch {$commands.ok -default 1}

        # pack this crud in...
        pack $commands -side bottom -fill x -expand n
        pack $simple -side top -fill both -ipady 20 -ipadx 20 -padx 5 -pady 5

        pack $options -side top -fill both -ipady 5 -ipadx 5 -padx 5 -pady 5

        bind $w(newDiffPopup) <Return> [list $commands.ok invoke]
        bind $w(newDiffPopup) <Escape> [list $commands.cancel invoke]

    }
    if {[winfo exists .client]} {
      centerWindow $w(newDiffPopup)
    } else {
      update
    }
    wm deiconify $w(newDiffPopup)
    raise $w(newDiffPopup)
    focus $w(newDiffPopup,entry1)
    tkwait variable waitvar
    wm withdraw $w(newDiffPopup)
}

proc open-more-options {} {
    global w

    grid $w(newDiffPopup).options.ml -row 0 -column 1 -sticky e
    grid $w(newDiffPopup).options.me -row 0 -column 2 -sticky nsew -pady 4
    grid $w(newDiffPopup).options.al -row 1 -column 1 -sticky e
    grid $w(newDiffPopup).options.ae -row 1 -column 2 -sticky nsew -pady 4
    grid $w(newDiffPopup).options.l1l -row 2 -column 1 -sticky e
    grid $w(newDiffPopup).options.l1e -row 2 -column 2 -sticky nsew -pady 4
    grid $w(newDiffPopup).options.l2l -row 3 -column 1 -sticky e
    grid $w(newDiffPopup).options.l2e -row 3 -column 2 -sticky nsew -pady 4

    $w(newDiffPopup).options.more configure -text "Less" \
      -command close-more-options
    set x [winfo width $w(newDiffPopup)]
    set y [winfo height $w(newDiffPopup)]
    set yi [winfo reqheight $w(newDiffPopup).options]
    set newy [expr $y + $yi]
    if {[winfo exists .client]} {
       centerWindow $w(newDiffPopup)
    } else {
       update
    }
}

proc close-more-options {} {
    global w
    global finfo

    grid remove $w(newDiffPopup).options.ml
    grid remove $w(newDiffPopup).options.me
    grid remove $w(newDiffPopup).options.al
    grid remove $w(newDiffPopup).options.ae
    grid remove $w(newDiffPopup).options.l1l
    grid remove $w(newDiffPopup).options.l1e
    grid remove $w(newDiffPopup).options.l2l
    grid remove $w(newDiffPopup).options.l2e

    set g(mergefileset) ""
    set g(conflictset) ""
    set g(ancfileset) ""
    set g(ancfile) ""
    set finfo(userlbl,1) ""
    set finfo(userlbl,2) ""

    $w(newDiffPopup).options.more configure -text "More" \
      -command open-more-options
}

###############################################################################
# File browser for the "New Diff" dialog
###############################################################################
proc newDiffBrowse {title widget} {
    global w

    set foo [$widget get]
    set initialdir [file dirname $foo]
    set initialfile [file tail $foo]
    set filename [tk_getOpenFile -title $title -initialfile $initialfile \
      -initialdir $initialdir]
    if {[string length $filename] > 0} {
        $widget delete 0 end
        $widget insert 0 $filename
        $widget selection range 0 end
        $widget xview end
        focus $widget
        return $filename
    } else {
        after idle {raise $w(newDiffPopup)}
        return {}
    }
}

###############################################################################
# all the code to handle the report writing dialog.
###############################################################################
proc write-report {command args} {
    global g
    global w
    global report
    global finfo

    set w(reportPopup) .reportPopup
    switch -- $command {
    popup {
            if {![winfo exists $w(reportPopup)]} {
                write-report build
            }
            set report(filename) [file join [pwd] $report(filename)]
            write-report update

            centerWindow $w(reportPopup)
            wm deiconify $w(reportPopup)
            raise $w(reportPopup)
        }
    cancel {
            wm withdraw $w(reportPopup)
        }
    update {

            set stateLeft "disabled"
            set stateRight "disabled"
            if {$report(doSideLeft)} {
                set stateLeft "normal"
            }
            if {$report(doSideRight)} {
                set stateRight "normal"
            }

            $w(reportLinenumLeft) configure -state $stateLeft
            $w(reportCMLeft) configure -state $stateLeft
            $w(reportTextLeft) configure -state $stateLeft

            $w(reportLinenumRight) configure -state $stateRight
            $w(reportCMRight) configure -state $stateRight
            $w(reportTextRight) configure -state $stateRight

        }
    save {
            set leftLines [lindex [split [$w(LeftText) index end-1lines] .] 0]
            set rightLines [lindex [split [$w(RightText) index end-1lines] .] 0]

            # number of lines of the largest window...
            set maxlines [max $leftLines $rightLines]

            # probably ought to catch this, in case it fails. Maybe later...
            set handle [open $report(filename) w]

            puts $handle "$g(name) $g(version) report"

            # write the file names
            if {$report(doSideLeft) == 1 && $report(doSideRight) == 1} {
                puts $handle "\nFile A: $finfo(lbl,1)\nFile B:  $finfo(lbl,2)\n"
            } elseif {$report(doSideLeft) == 1} {
                puts $handle "\nFile: $finfo(lbl,1)"
            } else {
                puts $handle "\nFile: $finfo(lbl,2)"
            }

            puts $handle "number of diffs: $g(count)"

            set acount [set ccount [set dcount 0]]
            for {set i 1} {$i <= $g(count)} {incr i} {
                foreach {line s1 e1 s2 e2 type} $g(scrdiff,$i) { }
                switch -- $type {
                "d" {
                        incr dcount
                    }
                "a" {
                        incr acount
                    }
                "c" {
                        incr ccount
                    }
                }
            }

            puts $handle [format "    %6d regions were deleted" $dcount]
            puts $handle [format "    %6d regions were added" $acount]
            puts $handle [format "    %6d regions were changed" $ccount]

            puts $handle "\n"
            for {set i 1} {$i <= $maxlines} {incr i} {
                set out(Left) [set out(Right) ""]
                foreach side {Left Right} {

                    if {$side == "Left" && $i > $leftLines} break

                    if {$side == "Right" && $i > $rightLines} break


                    if {$report(doLineNumbers$side)} {
                        set widget $w(${side}Info)
                        set number [string trimright [$widget get "$i.0" \
                          "$i.0 lineend"]]

                        append out($side) [format "%6s " $number]
                    }

                    if {$report(doChangeMarkers$side)} {
                        set widget $w(${side}CB)
                        set data [$widget get "$i.0" "$i.1"]
                        append out($side) "$data "
                    }

                    if {$report(doText$side)} {
                        set widget $w(${side}Text)
                        append out($side) [string trimright [$widget get \
                          "$i.0" "$i.0 lineend"]]
                    }
                }

                if {$report(doSideLeft) == 1 && $report(doSideRight) == 1} {
                    set output [format "%-90s%-90s" $out(Left) $out(Right)]

                } elseif {$report(doSideRight) == 1} {
                    set output $out(Right)

                } elseif {$report(doSideLeft) == 1} {
                    set output $out(Left)

                } else {
                    # what a wasted effort!
                    set output ""
                }
                puts $handle [string trimright $output]
            }
            close $handle

            wm withdraw $w(reportPopup)
        }
    browse {
            set types {
                {{All Files}         {*}}
            }

            set path [tk_getSaveFile -defaultextension "" -filetypes $types \
              -initialfile $report(filename)]

            if {[string length $path] > 0} {
                set report(filename) $path
            }
        }
    build {
            catch {destroy $w(reportPopup)}
            toplevel $w(reportPopup)
            wm group $w(reportPopup) .
            wm transient $w(reportPopup) .
            wm title $w(reportPopup) "$g(name) - Generate Report"
            wm protocol $w(reportPopup) WM_DELETE_WINDOW [list write-report \
              cancel]
            wm withdraw $w(reportPopup)

            if {$g(windowingSystem) == "aqua"} {
                setAquaDialogStyle $w(reportPopup)
            }

            set cf [frame $w(reportPopup).clientFrame -bd 2 -relief groove]
            set bf [frame $w(reportPopup).buttonFrame -bd 0]
            pack $cf -side top -fill both -expand y -padx 5 -pady 5
            pack $bf -side bottom -fill x -expand n

            # buttons...
            set w(reportSave) $bf.save
            set w(reportCancel) $bf.cancel

            button $w(reportSave) -text "Save" -underline 0 -command \
              [list write-report save] -width 6
            button $w(reportCancel) -text "Cancel" -underline 0 \
              -command [list write-report cancel] -width 6

            pack $w(reportCancel) -side right -pady 5 -padx 5
            pack $w(reportSave) -side right -pady 5

            # client area.
            set col(Left) 0
            set col(Right) 1
            foreach side [list Left Right] {
                set choose [checkbutton $cf.choose$side]
                set linenum [checkbutton $cf.linenum$side]
                set cm [checkbutton $cf.changemarkers$side]
                set text [checkbutton $cf.text$side]

                $choose configure -text "$side Side" \
                  -variable report(doSide$side) -command [list write-report \
                  update]

                $linenum configure -text "Line Numbers" \
                  -variable report(doLineNumbers$side)
                $cm configure -text "Change Markers" \
                  -variable report(doChangeMarkers$side)
                $text configure -text "Text" -variable report(doText$side)

                grid $choose -row 0 -column $col($side) -sticky w
                grid $linenum -row 1 -column $col($side) -sticky w -padx 10
                grid $cm -row 2 -column $col($side) -sticky w -padx 10
                grid $text -row 3 -column $col($side) -sticky w -padx 10

                # save the widget paths for later use...
                set w(reportChoose$side) $choose
                set w(reportLinenum$side) $linenum
                set w(reportCM$side) $cm
                set w(reportText$side) $text
            }

            # the entry, label and button for the filename will get
            # stuffed into a frame for convenience...
            frame $cf.fileFrame -bd 0
            grid $cf.fileFrame -row 4 -columnspan 2 -sticky ew -padx 5

            label $cf.fileFrame.l -text "File:"
            entry $cf.fileFrame.e -textvariable report(filename) -width 30
            button $cf.fileFrame.b -text "Browse..." -pady 0 \
              -highlightthickness 0 -borderwidth 1 -command \
              [list write-report browse]

            pack $cf.fileFrame.l -side left -pady 4
            pack $cf.fileFrame.b -side right -pady 4 -padx 2
            pack $cf.fileFrame.e -side left -fill x -expand y -pady 4

            grid rowconfigure $cf 0 -weight 0
            grid rowconfigure $cf 1 -weight 0
            grid rowconfigure $cf 2 -weight 0
            grid rowconfigure $cf 3 -weight 0

            grid columnconfigure $cf 0 -weight 1
            grid columnconfigure $cf 1 -weight 1

            # make sure the widgets are in the proper state
            write-report update
        }
    }
}

###############################################################################
# Throw up an "about" window.
###############################################################################
proc do-about {} {
    global g

    set title "About $g(name)"
    set text {
<hdr>$g(name) $g(version)</hdr>

<itl>$g(name)</itl> is a Tcl/Tk front-end to <itl>diff</itl> for Unix and \
      Windows, and is Copyright (C) 1994-2005 by John M. Klassa.

Many of the toolbar icons were created by Dean S. Jones and used with his \
      permission. The icons have the following copyright:

Copyright(C) 1998 by Dean S. Jones
dean@gallant.com
http://www.gallant.com/icons.htm
http://www.javalobby.org/jfa/projects/icons/

<bld>This program is free software; you can redistribute it and/or modify it \
      under the terms of the GNU General Public License as published by the \
      Free Software Foundation; either version 2 of the License, or (at your \
      option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT \
      ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or \
      FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License \
      for more details.

You should have received a copy of the GNU General Public License along with \
      this program; if not, write to the Free Software Foundation, Inc., 59 \
      Temple Place, Suite 330, Boston, MA 02111-1307 USA</bld>
    }

    set text [subst -nobackslashes -nocommands $text]
    do-text-info .about $title $text
}

###############################################################################
# Throw up a "command line usage" window.
###############################################################################
proc do-usage {mode} {
    global g

    debug-info "do-usage ($mode)"
    set usage {
    $g(name) may be started in any of the following ways:

    Interactive selection of files to compare:
     	tkdiff

    Plain files:
     	tkdiff FILE1 FILE2

    Plain file with conflict markers:
     	tkdiff -conflict FILE

    Source control (AccuRev, BitKeeper, CVS, Subversion, Perforce, PVCS,
      RCS, SCCS, ClearCase)
     	tkdiff FILE
	tkdiff -rREV FILE
	tkdiff -rREV1 -rREV2 FILE
        tkdiff OLD-URL[@OLDREV] NEW-URL[@NEWREV] (Subversion)

    Additional optional parameters:
	-a ANCESTORFILE 
	-o MERGEOUTPUTFILE 
	-L LEFT_FILE_LABEL [-L RIGHT_FILE_LABEL]
    }

    set usage [subst -nobackslashes -nocommands $usage]

    set text {
$g(name) detects and supports RCS, CVS, Subversion and SCCS by looking for a \
      directory with the same name. It detects and supports PVCS by looking \
      for a vcs.cfg file. It detects and supports AccuRev, Perforce and \
      ClearCase by looking for the environment variables named ACCUREV_BIN, \
      P4CLIENT, and CLEARCASE_ROOT respectively.

In the first form, tkdiff will present a dialog to allow you to choose the \
      files to diff interactively. At present this dialog only supports a \
      diff between two files that already exist. There is no way to do a diff \
      against a file under souce code control (unless the previous versions \
      can be found on disk via a standard file selection dialog).

In the second form, at least one of the arguments must be the name of a plain \
      text file.  Symbolic links are acceptable, but at least one of the \
      filename arguments must point to a real file rather than to a directory.

In the remaining forms, <cmp>REV</cmp> (or <cmp>REV1</cmp> and \
      <cmp>REV2</cmp>) must be a valid revision number for <cmp>FILE</cmp>. \
      Where AccuRev, RCS, CVS, Subversion, SCCS, PVCS or Perforce is implied \
      but no revision number is specified, <cmp>FILE</cmp> is compared with \
      the the revision most recently checked in.

To merge a file with conflict markers generated by "<cmp>merge</cmp>", \
      "<cmp>cvs</cmp>", or "<cmp>vmrg</cmp>", use \
      "<cmp>tkdiff -conflict FILE</cmp>". The file is split into two temporary \
      files which you can merge as usual (see below).

For "<cmp>tkdiff FILE</cmp>" The CVS version has priority, followed by the \
      Subversion version, followed by the SCCS version -- i.e. if a CVS \
      directory is present, CVS; if not and a Subversion directory is \
      present, Subversion; if not and an SCCS directory is present, SCCS is \
      assumed; otherwise, if a CVS.CFG file is found, PVCS is assumed; \
      otherwise RCS is assumed. If none of the above apply and the AccuRev \
      environment variable ACCUREV_BIN is found, AccuRev is used. If P4CLIENT \
      is found, Perforce is used. If CLEARCASE_ROOT is found, ClearCase is used.

If the merge output filename is not specified, tkdiff will present a dialog \
      to allow you to choose the name of the merge output file.

Note further that anything with a leading dash that isn\'t recognized as a \
      $g(name) option is passed through to diff.  This permits you to \
      temporarily alter the way diff is called, without resorting to a change \
      in your preferences file.
}

    if {$mode == "cline"} {
        puts $usage
        exit 0
    }
    set text [subst -nobackslashes -nocommands $text]
    append usage $text
    do-text-info .usage "$g(name) Usage" $usage
}

###############################################################################
# Throw up a help window.
###############################################################################
proc do-help {} {
    global g

    set title "How to use the $g(name) GUI"
    set text {
<hdr>Layout</hdr>

The top row contains the File, Edit, View, Mark, Merge and Help menus. The \
      second row contains the labels which identify the contents of each text \
      window. Below that is a toolbar which contains\
      navigation and merge selection tools.

The left-most text widget displays the contents of <cmp>FILE1</cmp>, the most \
      recently checked-in revision, <cmp>REV</cmp> or <cmp>REV1</cmp>, \
      respectively (as per the startup options described in\
      the "On Command Line" help). The right-most widget displays the \
      contents of <cmp>FILE2</cmp>, <cmp>FILE</cmp> or <cmp>REV2</cmp>, \
      respectively. Clicking the right mouse button over either of\
      these windows will give you a context sensitive menu with actions that \
      will act on the window you clicked over. For example, if you click \
      right over the right hand window and select\
      "Edit", the file displayed on the right hand side will be loaded into a \
      text editor.

At the bottom of the display is a two line window called the \
      "Line Comparison" window. This will show the "current line" from the \
      left and right windows, one on top of the other. The "current line"\
      is defined by the line that has the blinking insertion cursor, which \
      can be set by merely clicking on any line in the display. This window \
      may be hidden if the <btn>View</btn> menu item\
      <btn>Show Line Comparison</btn> is deselected.
All difference regions (DRs) are highlighted to set them apart from the \
      surrounding text. The <itl>current difference region</itl>, or \
      <bld>CDR</bld>, is further set apart so that it can be\
      correlated to its partner in the other text widget (that is, the CDR on \
      the left matches the CDR on the right).

<hdr>Changing the CDR</hdr>

The CDR can be changed in a sequential manner by means of the <btn>Next</btn> \
      and <btn>Previous</btn> buttons. The <btn>First</btn> and \
      <btn>Last</btn> buttons allow you to quickly navigate to the\
      first or last CDR, respectively. For random access to the DRs, use the \
      dropdown listbox in the toolbar or the diff map, described below.

By clicking right over a window and using the popup menu you can select \
      <btn>Find Nearest Diff</btn> to find the diff record nearest the point \
      where you clicked.

You may also select any highlighted diff region as the current diff region by \
      double-clicking on it.

<hdr>Operations</hdr>

1. From the <btn>File</btn> menu:

The <btn>New...</btn> button displays a dialog where you may choose two files \
      to compare. Selecting "Ok" from the dialog will diff the two files. The \
      <btn>Recompute Diffs</btn> button recomputes the\
      differences between the two files whose names appear at the top of the \
      <itl>$g(name)</itl> window. The <btn>Write Report...</btn> lets you \
      create a report file that contains the information\
      visible in the windows. Lastly, the <btn>Exit</btn> button terminates \
      <itl>$g(name)</itl>.

2. From the <btn>Edit</btn> menu:

<btn>Copy</btn> copies the currently selected text to the system clipboard. \
      <btn>Find</btn> pops up a dialog to let you search either text window \
      for a specified text string. <btn>Edit File 1</btn> and <btn>Edit File \
      2</btn> launch an editor on the files displayed in the left- and \
      right-hand panes.  <btn>Preferences</btn> pops up a dialog box from \
      which display (and other) options can be
changed and saved.

3. From the <btn>View</btn> menu:

<btn>Show Line Numbers</btn> toggles the display of line numbers in the text \
      widgets. If <btn>Synchronize Scrollbars</btn> is on, the left and right \
      text widgets are synchronized i.e. scrolling one\
      of the windows scrolls the other. If <btn>Auto Center</btn> is on, \
      pressing the Next or Prev buttons centers the new CDR automatically. \
      <btn>Show Diff Map</btn> toggles the display of the diff\
      map (see below) on or off. <btn>Show Merge Preview</btn> shows or hides \
      the merge preview (see below). <btn>Show Line Comparison</btn> toggles \
      the display of the "line comparison" window at\
      the bottom of the display.

4. From the <btn>Mark</btn> menu:

The <btn>Mark Current Diff</btn> creates a new toolbar button that will jump \
      to the current diff region. The <btn>Clear Current Diff Mark</btn> will \
      remove the toolbar mark button associated with\
      the current diff region, if one exists.

5. From the <btn>Merge</btn> menu:

The <btn>Show Merge Window</btn> button pops up a window with the current \
      merged version of the two files. The <btn>Write Merge File</btn> button \
      will allow you to save the contents of that window\
      to a file.

6. From the <btn>Help</btn> menu:

The <btn>About $g(name)</btn> button displays copyright and author \
      information. The <btn>On GUI</btn> button generates this window. The \
      <btn>On Command Line</btn> button displays help on the\
      $g(name) command line options. The <btn>On Preferences</btn> button \
      displays help on the user-settable preferences.

7. From the toolbar:

The first tool is a dropdown list of all of the differences in a standard \
      diff-type format. You may use this list to go directly to any diff \
      record. The <btn>Next</btn> and <btn>Previous</btn>\
      buttons take you to the "next" and "previous" DR, respectively. The \
      <btn>First</btn> and <btn>Last</btn> buttons take you to the \
      "first" and "last" DR. The <btn>Center</btn> button centers the\
      CDRs in their respective text windows. You can set <btn>Auto \
      Center</btn> in <btn>Preferences</btn> to do this automatically for you \
      as you navigate through the diff records.

<hdr>Keyboard Navigation</hdr>

When a text widget has the focus, you may use the following shortcut keys:
<cmp>
	f	First diff
	c	Center current diff
	l	Last diff
	n	Next diff
	p	Previous diff
	1	Merge Choice 1
	2	Merge Choice 2
</cmp>
The cursor, Home, End, PageUp and PageDown keys work as expected, adjusting \
      the view in whichever text window has the focus. Note that if \
      <btn>Synchronize Scrollbars</btn> is set in\
      <btn>Preferences</btn>, both windows will scroll at the same time.

<hdr>Scrolling</hdr>

To scroll the text widgets independently, make sure <btn>Synchronize \
      Scrollbars</btn> in <btn>Preferences</btn> is off. If it is on, \
      scrolling any text widget scrolls all others. Scrolling does not\
      change the current diff record (CDR).

<hdr>Diff Marks</hdr>

You can set "markers" at specific diff regions for easier navigation. To do \
      this, click on the <btn>Set Mark</btn> button. It will create a new \
      toolbar button that will jump back to this diff\
      region. To clear a diff mark, go to that diff record and click on the \
      <btn>Clear Mark</btn> button.

<hdr>Diff Map</hdr>

The diff map is a map of all the diff regions. It is shown in the middle of \
      the main window if "Diff Map" on the View menu is on. The map is a \
      miniature of the file's diff regions from top to\
      bottom. Each diff region is rendered as a patch of color, Delete as \
      red, Insert as green and Change as blue. In the case of a 3-way merge, \
      overlap regions are marked in yellow. The height of each patch \
      corresponds to the relative size of the diff region. A\
      thumb lets you interact with the map as if it were a scrollbar.
All diff regions are drawn on the map even if too small to be visible. For \
      large files with small diff regions, this may result in patches \
      overwriting each other.

<hdr>Merging</hdr>

To merge the two files, go through the difference regions (via "Next", \
      "Prev" or whatever other means you prefer) and select "Left" or \
      "Right" (next to the "Merge Choice:" label) for each. Selecting\
      "Left" means that the the left-most file's version of the difference \
      will be used in creating the final result; choosing "Right" means that \
      the right-most file's difference will be used. Each\
      choice is recorded, and can be changed arbitrarily many times. To \
      commit the final, merged result to disk, choose "Write Merge File..." \
      from the <btn>Merge</btn> menu.

<hdr>Merge Preview</hdr>

To see a preview of the file that would be written by "Write Merge File...", \
      select "Show Merge Window" in the View menu. A separate window is shown \
      containing the preview. It is updated as you\
      change merge choices. It is synchronized with the other text widgets if \
      "Synchronize Scrollbars" is on.

<hdr>Author</hdr>
John M. Klassa

<hdr>Comments</hdr>
Questions and comments should be sent to the TkDiff mailing list at \
      tkdiff-discuss@lists.sourceforge.net.
    }

    set text [subst -nobackslashes -nocommands $text]
    do-text-info .help $title $text
}

######################################################################
# display help on the preferences
######################################################################
proc do-help-preferences {} {
    global g
    global pref

    customize-initLabels

    set title "$g(name) Preferences"
    set text {
<hdr>Overview</hdr>

Preferences are stored in a file in your home directory (identified by the \
      environment variable <cmp>HOME</cmp>.) If the environment variable \
      <cmp>HOME</cmp> is not set the platform-specific variant\
      of "/" will be used. If you are on a Windows platform the file will be \
      named <cmp>_tkdiff.rc</cmp> and will have the attribute "hidden". For \
      all other platforms the file will be named\
      ".tkdiffrc". You may override the name and location of this file by \
      setting the environment variable <cmp>TKDIFFRC</cmp> to whatever \
      filename you wish.

Preferences are organized into three categories: General, Display and \
      Appearance.

<hdr>General</hdr>

<bld>$pref(diffcmd)</bld>

This is the command to run to generate a diff of the two files. Typically \
      this will be "diff".  When this command is run, the ignore-blanks \
      options and the names of two files to be diffed will be added as the \
      last to arguments on the command line.

<bld>$pref(ignoreblanksopt)</bld>

Arguments to send with the diff command to tell it how to ignore whitespace. \
      If you are using gnu diff, "-b" or "--ignore-space-change" ignores \
      changes in the amount of whitespace, while "-w" or \
      "--ignore-all-space" ignores all white space.

<bld>$pref(tmpdir)</bld>

The name of a directory for files that are temporarily created while $g(name) \
      is running.

<bld>$pref(editor)</bld>

The name of an external editor program to use when editing a file (ie: when \
      you select "Edit" from the popup menu). If this value is blank, a \
      simple editor built in to $g(name) will be used. For\
      windows users you might want to set this to "notepad". Unix users may \
      want to set this to "xterm -e vi" or perhaps "gnuclient". When run, the \
      name of the file to edit will be appened as the\
      last argument on the command line.
If the supplied string contains the string "\$file", it\'s treated as a whole \
      command line, where the following parameters can be used:
 \$file: the file of your choice
 \$line: the starting line of the current diff
For example, in the case of NEdit or Emacs you can use "nc -line \$line \
     \$file" and "emacs +\$line \$file" respectively.

<bld>$pref(geometry)</bld>

This defines the default size, in characters of the two text windows. The \
      format should be <cmp>WIDTHxHEIGHT</cmp>. For example, "80x40".

<bld>$pref(fancyButtons)</bld>

If set, toolbar buttons will mimic the visual behavior of typical Microsoft \
      Windows applications. Buttons will initially be flat until the cursor \
      moves over them, at which time they will be raised.
If unset, toolbar buttons will always appear raised.
This feature is not supported in MacOSX.

<bld>$pref(toolbarIcons)</bld>

If set, the toolbar buttons will use icons instead of text labels.
If unset, the toolbar buttons will use text labels instead of icons.

<bld>$pref(autocenter)</bld>

If set, whenever a new diff record becomes the current diff record (for \
      example, when pressing the next or previous buttons), the diff record \
      will be automatically centered on the screen.
If unset, no automatic scrolling will occur.

<bld>$pref(syncscroll)</bld>

If set, scrolling either text window will result in both windows scrolling.
If not set, the windows will scroll independent of each other.

<bld>$pref(autoselect)</bld>

If set, automatically select the nearest visible diff region when scrolling.
If not set, the current diff region will not change during scrolling.
This only takes effect if <bld>$pref(syncscroll)</bld> is set.

<hdr>Display</hdr>

<bld>$pref(showln)</bld>

If set, line numbers will be displayed alongside each line of each file.
If not set, no line numbers will appear.

<bld>$pref(tagln)</bld>

If set, line numbers are highlighted with the options defined in the \
      Appearance section of the preferences.
If not set, line numbers won\'t be highlighted.

<bld>$pref(showcbs)</bld>

If set, change bars will be displayed alongside each line of each file.
If not set, no change bars will appear.

<bld>$pref(tagcbs)</bld>

If set, change indicators will be highlighted. If <itl>$pref(colorcbs)</itl> \
      is set they will appear as solid colored bars that match the colors \
      used in the diff map. If <itl>$pref(colorcbs)</itl>\
      is not set, the change indicators will be highlighted according to the \
      options defined in the Appearance section of preferences.

<bld>$pref(showmap)</bld>

If set, colorized, graphical "diff map" will be displayed between the two \
      files, showing regions that have changed. Red is used to show deleted \
      lines, green for added lines, blue for changed\
      lines, and yellow for overlapping lines during a 3-way merge.
If not set, the diff map will not be shown.

<bld>$pref(showlineview)</bld>

If set, show a window at the bottom of the display that shows the current \
      line from each file, one on top of the other. This window is most \
      useful to do a byte-by-byte comparison of a line that has\
      changed.
If not set, the window will not be shown.

<bld>$pref(showinline1)</bld>

If set, show inline diffs in the main window. This is useful to see what the \
      actual diffs are within a large diff region. \
If not set, the inline diffs are neither computed nor shown.  This is the \
      simpler approach, where byte-by-byte comparisons \
are used.

<bld>$pref(showinline2)</bld>

If set, show inline diffs in the main window. This is useful to see what the \
      actual diffs are within a large diff region. \
If not set, the inline diffs are neither computed nor shown.  This approach \
      is more complex, but should give more pleasing \
results for source code and written text files.  This is the \
      Ratcliff/Obershelp pattern matching algorithm which recursively \
finds the largest common substring, and recursively repeats on the left and \
      right remainders.

<bld>$pref(tagtext)</bld>

If set, the file contents will be highlighted with the options defined in the \
      Appearance section of the preferences.
If not set, the file contents won\'t be highlighted.

<bld>$pref(colorcbs)</bld>

If set, the change bars will display as solid bars of color that match the \
      colors used by the diff map.
If not set, the change bars will display a "+" for lines that exist in only \
      one file, a "-" for lines that are missing from only one file, and \
      "!" for lines that are different between the two files.

<hdr>Appearance</hdr>

<bld>$pref(textopt)</bld>

This is a list of Tk text widget options that are applied to each of the two \
      text windows in the main display. If you have Tk installed on your \
      machine these will be documented in the "Text.n" man\
      page.

<bld>$pref(difftag)</bld>

This is a list of Tk text widget tag options that are applied to all diff \
      regions. Use this option to make diff regions stand out from regular text.

<bld>$pref(deltag)</bld>

This is a list of Tk text widget tag options that are applied to the current \
      diff region. These options have a higher priority than those for all \
      diff regions. So, for example, if you set the\
      forground for all diff regions to be black and set the foreground for \
      the current diff region to be blue, the current diff region foreground \
      color will be used.

<bld>$pref(instag)</bld>

This is a list of Tk text widget tag options that are applied to regions that \
      have been inserted. These options have a higher priority than those for \
      all diff regions.

<bld>$pref(chgtag)</bld>

This is a list of Tk text widget tag options that are applied to regions that \
      have been changed. These options have a higher priority than those for \
      all diff regions.

<bld>$pref(currtag)</bld>

This is a list of Tk text widget tag options that are applied to the current \
      diff region. These tags have a higher priority than those for all diff \
      regions, and a higher priority than the change,\
      inserted and deleted diff regions.

<bld>$pref(inlinetag)</bld>

This is a list of Tk text widget tag options that are applied to differences \
      within lines in a diff region. These tags have a higher priority than \
      those for all diff regions, and a higher priority than the change,\
      inserted and deleted diff regions.

<bld>$pref(bytetag)</bld>

This is a list of Tk text widget tag options that are applied to individual \
      characters in the line view. These options do not affect the main text \
      displays.

<bld>$pref(tabstops)</bld>

This defines the number of characters for each tabstop in the main display \
      windows. The default is 8.
    }

    # since we have embedded references to the preference labels in
    # the text, we need to perform substitutions. Because of this, if
    # you edit the above text, be sure to properly escape any dollar
    # signs that are not meant to be treated as a variable reference

    set text [subst -nocommands $text]
    do-text-info .help-preferences $title $text
}

######################################################################
#
# text formatting routines derived from Klondike
# Reproduced here with permission from their author.
#
# Copyright (C) 1993,1994 by John Heidemann <johnh@ficus.cs.ucla.edu>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of John Heidemann may not be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY JOHN HEIDEMANN ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL JOHN HEIDEMANN BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
######################################################################
proc put-text {tw txt} {

    $tw configure -font {Fixed 12}

    $tw configure -font -*-Times-Medium-R-Normal-*-14-*

    $tw tag configure bld -font -*-Times-Bold-R-Normal-*-14-*
    $tw tag configure cmp -font -*-Courier-Medium-R-Normal-*-12-*
    $tw tag configure hdr -font -*-Helvetica-Bold-R-Normal-*-16-* -underline 1
    $tw tag configure itl -font -*-Times-Medium-I-Normal-*-14-*
    $tw tag configure ttl -font -*-Helvetica-Bold-R-Normal-*-18-*
    #$tw tag configure h3 -font -*-Helvetica-Bold-R-Normal-*-14-*
    #$tw tag configure itl -font -*-Times-Medium-I-Normal-*-14-*
    #$tw tag configure rev -foreground white -background black


    $tw mark set insert 0.0

    set t $txt

    while {[regexp -indices {<([^@>]*)>} $t match inds] == 1} {

        set start [lindex $inds 0]
        set end [lindex $inds 1]
        set keyword [string range $t $start $end]

        set oldend [$tw index end]

        $tw insert end [string range $t 0 [expr {$start - 2}]]

        purge-all-tags $tw $oldend insert

        if {[string range $keyword 0 0] == "/"} {
            set keyword [string trimleft $keyword "/"]
            if {[info exists tags($keyword)] == 0} {
                error "end tag $keyword without beginning"
            }
            $tw tag add $keyword $tags($keyword) insert
            unset tags($keyword)
        } else {
            if {[info exists tags($keyword)] == 1} {
                error "nesting of begin tag $keyword"
            }
            set tags($keyword) [$tw index insert]
        }

        set t [string range $t [expr {$end + 2}] end]
    }

    set oldend [$tw index end]
    $tw insert end $t
    purge-all-tags $tw $oldend insert
}

proc purge-all-tags {w start end} {
    foreach tag [$w tag names $start] {
        $w tag remove $tag $start $end
    }
}

# Open one of the diffed files in an editor if possible
proc do-edit {} {
    global g
    global opts
    global finfo
    global w

    if {$g(activeWindow) == $w(LeftText)} {
        set fileno 1
    } elseif {$g(activeWindow) == $w(RightText)} {
        set fileno 2
    } else {
        set fileno 1
    }

    if {$finfo(tmp,$fileno)} {
        do-error "This file is not editable"
    } else {
        if {[string length [string trim $opts(editor)]] == 0} {
            simpleEd open $finfo(pth,$fileno)
        } elseif {[regexp "\\\$file" "$opts(editor)"] == 1} {
            set line [lindex [extract $g(currdiff)] [expr {($fileno-1) *2+1}]]
            set file $finfo(pth,$fileno)
            eval set commandline \"$opts(editor) &\"
            eval exec $commandline
        } else {
            eval exec $opts(editor) "{$finfo(pth,$fileno)}" &
        }
    }
}

##########################################################################
# platform-specific stuff
##########################################################################
proc setAquaDialogStyle {toplevel} {
    tk::unsupported::MacWindowStyle style $toplevel movableDBoxProc
}

##########################################################################
# A simple editor, from Bryan Oakley.
##########################################################################
proc simpleEd {command args} {
    global textfont

    switch -- $command {
    open {
            set filename [lindex $args 0]

            set w .editor
            set count 0
            while {[winfo exists ${w}$count]} {
                incr count 1
            }
            set w ${w}$count

            toplevel $w -borderwidth 2 -relief sunken
            wm title $w "$filename - Simple Editor"
            wm group $w .

            menu $w.menubar
            $w configure -menu $w.menubar

            $w.menubar add cascade -label "File" -menu $w.menubar.fileMenu
            $w.menubar add cascade -label "Edit" -menu $w.menubar.editMenu

            menu $w.menubar.fileMenu
            menu $w.menubar.editMenu

            $w.menubar.fileMenu add command -label "Save" -underline 1 \
              -command [list simpleEd save $filename $w]
            $w.menubar.fileMenu add command -label "Save As..." -underline 1 \
              -command [list simpleEd saveAs $filename $w]
            $w.menubar.fileMenu add separator
            $w.menubar.fileMenu add command -label "Exit" -underline 1 \
              -command [list simpleEd exit $w]

            $w.menubar.editMenu add command -label "Cut" -command [list event \
              generate $w.text <<Cut>>]
            $w.menubar.editMenu add command -label "Copy" -command \
              [list event generate $w.text <<Copy>>]
            $w.menubar.editMenu add command -label "Paste" -command \
              [list event generate $w.text <<Paste>>]

            text $w.text -wrap none -xscrollcommand [list $w.hsb set] \
              -yscrollcommand [list $w.vsb set] -borderwidth 0 -font $textfont
            scrollbar $w.vsb -orient vertical -command [list $w.text yview]
            scrollbar $w.hsb -orient horizontal -command [list $w.text xview]

            grid $w.text -row 0 -column 0 -sticky nsew
            grid $w.vsb -row 0 -column 1 -sticky ns
            grid $w.hsb -row 1 -column 0 -sticky ew

            grid columnconfigure $w 0 -weight 1
            grid columnconfigure $w 1 -weight 0
            grid rowconfigure $w 0 -weight 1
            grid rowconfigure $w 1 -weight 0

            set fd [open $filename]
            $w.text insert 1.0 [read $fd]
            close $fd
        }
    save {
            set filename [lindex $args 0]
            set w [lindex $args 1]
            set fd [open $filename w]
            puts $fd [$w.text get 1.0 "end-1c"]
            close $fd
        }
    saveAs {
            set filename [lindex $args 0]
            set w [lindex $args 1]
            set filename [tk_getSaveFile -initialfile $filename]
            if {$filename != ""} {
                simpleEd save $filename $w
            }
        }
    exit {
            set w [lindex $args 0]
            destroy $w
        }
    }
}

# end of simpleEd

# Copyright (c) 1998-2005, Bryan Oakley
# All Rights Reservered
#
# Bryan Oakley
# oakley@bardo.clearlight.com
#
# combobox v2.2.2 September 22, 2002
#
# a combobox / dropdown listbox (pick your favorite name) widget
# written in pure tcl
#
# this code is freely distributable without restriction, but is
# provided as-is with no warranty expressed or implied.
#
# thanks to the following people who provided beta test support or
# patches to the code (in no particular order):
#
# Scott Beasley     Alexandre Ferrieux      Todd Helfter
# Matt Gushee       Laurent Duperval        John Jackson
# Fred Rapp         Christopher Nelson
# Eric Galluzzo     Jean-Francois Moine
#
# A special thanks to Martin M. Hunt who provided several good ideas,
# and always with a patch to implement them. Jean-Francois Moine,
# Todd Helfter and John Jackson were also kind enough to send in some
# code patches.
#
# ... and many others over the years.

package provide combobox 2.2.2

namespace eval ::combobox {

    # this is the public interface
    namespace export combobox

    # these contain references to available options
    variable widgetOptions

    # these contain references to available commands and subcommands
    variable widgetCommands
    variable scanCommands
    variable listCommands
}

# ::combobox::combobox --
#
#     This is the command that gets exported. It creates a new
#     combobox widget.
#
# Arguments:
#
#     w        path of new widget to create
#     args     additional option/value pairs (eg: -background white, etc.)
#
# Results:
#
#     It creates the widget and sets up all of the default bindings
#
# Returns:
#
#     The name of the newly create widget

proc ::combobox::combobox {w args} {
    variable widgetOptions
    variable widgetCommands
    variable scanCommands
    variable listCommands

    # perform a one time initialization
    if {![info exists widgetOptions]} {
        Init
    }

    # build it...
    eval Build $w $args

    # set some bindings...
    SetBindings $w

    # and we are done!
    return $w
}

# ::combobox::Init --
#
#     Initialize the namespace variables. This should only be called
#     once, immediately prior to creating the first instance of the
#     widget
#
# Arguments:
#
#    none
#
# Results:
#
#     All state variables are set to their default values; all of
#     the option database entries will exist.
#
# Returns:
#
#     empty string

proc ::combobox::Init {} {
    variable widgetOptions
    variable widgetCommands
    variable scanCommands
    variable listCommands
    variable defaultEntryCursor

    array set widgetOptions [list -background \
      {background          Background} -bd -borderwidth -bg -background \
      -borderwidth {borderWidth         BorderWidth} -command \
      {command Command} -commandstate {commandState        State} \
      -cursor {cursor              Cursor} \
      -disabledbackground {disabledBackground  DisabledBackground} \
      -disabledforeground {disabledForeground  DisabledForeground} \
      -dropdownwidth {dropdownWidth       DropdownWidth} -editable \
      {editable            Editable} -fg -foreground -font \
      {font                Font} -foreground {foreground          Foreground} \
      -height {height              Height} \
      -highlightbackground {highlightBackground HighlightBackground} \
      -highlightcolor {highlightColor      HighlightColor} \
      -highlightthickness {highlightThickness  HighlightThickness} \
      -image {image               Image} -maxheight \
      {maxHeight           Height} -opencommand {opencommand         Command} \
      -relief {relief              Relief} \
      -selectbackground {selectBackground    Foreground} \
      -selectborderwidth {selectBorderWidth   BorderWidth} \
      -selectforeground {selectForeground    Background} -state \
      {state               State} -takefocus {takeFocus           TakeFocus} \
      -textvariable {textVariable        Variable} -value \
      {value               Value} -width {width               Width} \
      -xscrollcommand {xScrollCommand      ScrollCommand}]


    set widgetCommands [list bbox cget configure curselection delete get \
      icursor index insert list scan selection xview select toggle open close]

    set listCommands [list delete get index insert size]

    set scanCommands [list mark dragto]

    # why check for the Tk package? This lets us be sourced into
    # an interpreter that doesn't have Tk loaded, such as the slave
    # interpreter used by pkg_mkIndex. In theory it should have no
    # side effects when run
    if {[lsearch -exact [package names] "Tk"] != -1} {

        ##################################################################
        #- this initializes the option database. Kinda gross, but it works
        #- (I think).
        ##################################################################

        # the image used for the button...
        if {$::tcl_platform(platform) == "windows"} {
            image create bitmap ::combobox::bimage -data {
                #define down_arrow_width 12
                #define down_arrow_height 12
                static char down_arrow_bits[] = {
                    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                    0xfc,0xf1,0xf8,0xf0,0x70,0xf0,0x20,0xf0,
                    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00;
                }
            }
        } else {
            image create bitmap ::combobox::bimage -data {
                #define down_arrow_width 15
                #define down_arrow_height 15
                static char down_arrow_bits[] = {
                    0x00,0x80,0x00,0x80,0x00,0x80,0x00,0x80,
                    0x00,0x80,0xf8,0x8f,0xf0,0x87,0xe0,0x83,
                    0xc0,0x81,0x80,0x80,0x00,0x80,0x00,0x80,
                    0x00,0x80,0x00,0x80,0x00,0x80
                }
            }
        }

        # compute a widget name we can use to create a temporary widget
        set tmpWidget ".__tmp__"
        set count 0
        while {[winfo exists $tmpWidget] == 1} {
            set tmpWidget ".__tmp__$count"
            incr count
        }

        # get the scrollbar width. Because we try to be clever and draw our
        # own button instead of using a tk widget, we need to know what size
        # button to create. This little hack tells us the width of a scroll
        # bar.
        #
        # NB: we need to be sure and pick a window  that doesn't already
        # exist...
        scrollbar $tmpWidget
        set sb_width [winfo reqwidth $tmpWidget]
        destroy $tmpWidget

        # steal options from the entry widget
        # we want darn near all options, so we'll go ahead and do
        # them all. No harm done in adding the one or two that we
        # don't use.
        entry $tmpWidget
        foreach foo [$tmpWidget configure] {
            # the cursor option is special, so we'll save it in
            # a special way
            if {[lindex $foo 0] == "-cursor"} {
                set defaultEntryCursor [lindex $foo 4]
            }
            if {[llength $foo] == 5} {
                set option [lindex $foo 1]
                set value [lindex $foo 4]
                option add *Combobox.$option $value widgetDefault

                # these options also apply to the dropdown listbox
                if {[string compare $option "foreground"] == 0 || \
                  [string compare $option "background"] == 0 || \
                  [string compare $option "font"] == 0} {
                    option add *Combobox*ComboboxListbox.$option $value \
                      widgetDefault
                }
            }
        }
        destroy $tmpWidget

        # these are unique to us...
        option add *Combobox.dropdownWidth {} widgetDefault
        option add *Combobox.openCommand {} widgetDefault
        option add *Combobox.cursor {} widgetDefault
        option add *Combobox.commandState normal widgetDefault
        option add *Combobox.editable 1 widgetDefault
        option add *Combobox.maxHeight 10 widgetDefault
        option add *Combobox.height 0
    }

    # set class bindings
    SetClassBindings
}

# ::combobox::SetClassBindings --
#
#    Sets up the default bindings for the widget class
#
#    this proc exists since it's The Right Thing To Do, but
#    I haven't had the time to figure out how to do all the
#    binding stuff on a class level. The main problem is that
#    the entry widget must have focus for the insertion cursor
#    to be visible. So, I either have to have the entry widget
#    have the Combobox bindtag, or do some fancy juggling of
#    events or some such. What a pain.
#
# Arguments:
#
#    none
#
# Returns:
#
#    empty string

proc ::combobox::SetClassBindings {} {

    # make sure we clean up after ourselves...
    bind Combobox <Destroy> [list ::combobox::DestroyHandler %W]

    # this will (hopefully) close (and lose the grab on) the
    # listbox if the user clicks anywhere outside of it. Note
    # that on Windows, you can click on some other app and
    # the listbox will still be there, because tcl won't see
    # that button click
    set this {[::combobox::convert %W -W]}
    bind Combobox <Any-ButtonPress> "$this close"
    bind Combobox <Any-ButtonRelease> "$this close"

    # this helps (but doesn't fully solve) focus issues. The general
    # idea is, whenever the frame gets focus it gets passed on to
    # the entry widget
    #bind Combobox <FocusIn> {::combobox::tkTabToWindow \
        #[::combobox::convert %W -W].entry}

    # this closes the listbox if we get hidden
    bind Combobox <Unmap> {[::combobox::convert %W -W] close}

    return ""
}

# ::combobox::SetBindings --
#
#    here's where we do most of the binding foo. I think there's probably
#    a few bindings I ought to add that I just haven't thought
#    about...
#
#    I'm not convinced these are the proper bindings. Ideally all
#    bindings should be on "Combobox", but because of my juggling of
#    bindtags I'm not convinced thats what I want to do. But, it all
#    seems to work, its just not as robust as it could be.
#
# Arguments:
#
#    w    widget pathname
#
# Returns:
#
#    empty string

proc ::combobox::SetBindings {w} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    # juggle the bindtags. The basic idea here is to associate the
    # widget name with the entry widget, so if a user does a bind
    # on the combobox it will get handled properly since it is
    # the entry widget that has keyboard focus.
    bindtags $widgets(entry) [concat $widgets(this) [bindtags $widgets(entry)]]

    bindtags $widgets(button) [concat $widgets(this) \
      [bindtags $widgets(button)]]

    # override the default bindings for tab and shift-tab. The
    # focus procs take a widget as their only parameter and we
    # want to make sure the right window gets used (for shift-
    # tab we want it to appear as if the event was generated
    # on the frame rather than the entry.
    #bind $widgets(entry) <Tab> "::combobox::tkTabToWindow \[tk_focusNext \
        #$widgets(entry)\]; break"
    #bind $widgets(entry) <Shift-Tab> \
        #"::combobox::tkTabToWindow \[tk_focusPrev $widgets(this)\]; break"

    # this makes our "button" (which is actually a label)
    # do the right thing
    bind $widgets(button) <ButtonPress-1> [list $widgets(this) toggle]

    # this lets the autoscan of the listbox work, even if they
    # move the cursor over the entry widget.
    bind $widgets(entry) <B1-Enter> "break"

    bind $widgets(listbox) <ButtonRelease-1> "::combobox::Select \
      [list $widgets(this)] \[$widgets(listbox) nearest %y\]; break"

    bind $widgets(vsb) <ButtonPress-1> {continue}
    bind $widgets(vsb) <ButtonRelease-1> {continue}

    bind $widgets(listbox) <Any-Motion> {
        %W selection clear 0 end
        %W activate @%x,%y
        %W selection anchor @%x,%y
        %W selection set @%x,%y @%x,%y
        # need to do a yview if the cursor goes off the top
        # or bottom of the window... (or do we?)
    }

    # these events need to be passed from the entry widget
    # to the listbox, or otherwise need some sort of special
    # handling.
    foreach event [list <Up> <Down> <Tab> <Return> <Escape> <Next> <Prior> \
      <Double-1> <1> <Any-KeyPress> <FocusIn> <FocusOut>] {
        bind $widgets(entry) $event [list ::combobox::HandleEvent \
          $widgets(this) $event]
    }

    # like the other events, <MouseWheel> needs to be passed from
    # the entry widget to the listbox. However, in this case we
    # need to add an additional parameter
    catch {
        bind $widgets(entry) <MouseWheel> [list ::combobox::HandleEvent \
          $widgets(this) <MouseWheel> %D]
    }
}

# ::combobox::Build --
#
#    This does all of the work necessary to create the basic
#    combobox.
#
# Arguments:
#
#    w        widget name
#    args     additional option/value pairs
#
# Results:
#
#    Creates a new widget with the given name. Also creates a new
#    namespace patterened after the widget name, as a child namespace
#    to ::combobox
#
# Returns:
#
#    the name of the widget

proc ::combobox::Build {w args} {
    variable widgetOptions

    if {[winfo exists $w]} {
        error "window name \"$w\" already exists"
    }

    # create the namespace for this instance, and define a few
    # variables
    namespace eval ::combobox::$w {

        variable ignoreTrace 0
        variable oldFocus {}
        variable oldGrab {}
        variable oldValue {}
        variable options
        variable this
        variable widgets

        set widgets(foo) foo ;# coerce into an array
        set options(foo) foo ;# coerce into an array

        unset widgets(foo)
        unset options(foo)
    }

    # import the widgets and options arrays into this proc so
    # we don't have to use fully qualified names, which is a
    # pain.
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    # this is our widget -- a frame of class Combobox. Naturally,
    # it will contain other widgets. We create it here because
    # we need it in order to set some default options.
    set widgets(this) [frame $w -class Combobox -takefocus 0]
    set widgets(entry) [entry $w.entry -takefocus 1]
    set widgets(button) [label $w.button -takefocus 0]

    # this defines all of the default options. We get the
    # values from the option database. Note that if an array
    # value is a list of length one it is an alias to another
    # option, so we just ignore it
    foreach name [array names widgetOptions] {
        if {[llength $widgetOptions($name)] == 1} continue

        set optName [lindex $widgetOptions($name) 0]
        set optClass [lindex $widgetOptions($name) 1]

        set value [option get $w $optName $optClass]
        set options($name) $value
    }

    # a couple options aren't available in earlier versions of
    # tcl, so we'll set them to sane values. For that matter, if
    # they exist but are empty, set them to sane values.
    if {[string length $options(-disabledforeground)] == 0} {
        set options(-disabledforeground) $options(-foreground)
    }
    if {[string length $options(-disabledbackground)] == 0} {
        set options(-disabledbackground) $options(-background)
    }

    # if -value is set to null, we'll remove it from our
    # local array. The assumption is, if the user sets it from
    # the option database, they will set it to something other
    # than null (since it's impossible to determine the difference
    # between a null value and no value at all).
    if {[info exists options(-value)] && [string length $options(-value)] == \
      0} {
        unset options(-value)
    }

    # we will later rename the frame's widget proc to be our
    # own custom widget proc. We need to keep track of this
    # new name, so we'll define and store it here...
    set widgets(frame) ::combobox::${w}::$w

    # gotta do this sooner or later. Might as well do it now
    pack $widgets(entry) -side left -fill both -expand yes
    pack $widgets(button) -side right -fill y -expand no

    # I should probably do this in a catch, but for now it's
    # good enough... What it does, obviously, is put all of
    # the option/values pairs into an array. Make them easier
    # to handle later on...
    array set options $args

    # now, the dropdown list... the same renaming nonsense
    # must go on here as well...
    set widgets(dropdown) [toplevel $w.top]
    set widgets(listbox) [listbox $w.top.list]
    set widgets(vsb) [scrollbar $w.top.vsb]

    pack $widgets(listbox) -side left -fill both -expand y

    # fine tune the widgets based on the options (and a few
    # arbitrary values...)

    # NB: we are going to use the frame to handle the relief
    # of the widget as a whole, so the entry widget will be
    # flat. This makes the button which drops down the list
    # to appear "inside" the entry widget.

    $widgets(vsb) configure -command "$widgets(listbox) yview" \
      -highlightthickness 0

    $widgets(button) configure -highlightthickness 0 -borderwidth 1 \
      -relief raised -width [expr {[winfo reqwidth $widgets(vsb)] - 2}]

    $widgets(entry) configure -borderwidth 0 -relief flat -highlightthickness 0

    $widgets(dropdown) configure -borderwidth 1 -relief sunken

    $widgets(listbox) configure -selectmode browse \
      -background [$widgets(entry) cget -bg] -yscrollcommand \
      "$widgets(vsb) set" -exportselection false -borderwidth 0


    # do some window management foo on the dropdown window
    # There seems to be some order dependency here on some platforms
    wm transient $widgets(dropdown) [winfo toplevel $w]
    wm group $widgets(dropdown) [winfo parent $w]
    wm resizable $widgets(dropdown) 0 0
    wm overrideredirect $widgets(dropdown) 1
    wm withdraw $widgets(dropdown)

    # this moves the original frame widget proc into our
    # namespace and gives it a handy name
    rename ::$w $widgets(frame)

    # now, create our widget proc. Obviously (?) it goes in
    # the global namespace. All combobox widgets will actually
    # share the same widget proc to cut down on the amount of
    # bloat.
    proc ::$w {command args} "eval ::combobox::WidgetProc $w \$command \$args"


    # ok, the thing exists... let's do a bit more configuration.
    if {[catch "::combobox::Configure [list $widgets(this)] [array get \
      options]" error]} {
        catch {destroy $w}
        error "internal error: $error"
    }

    return ""
}

# ::combobox::HandleEvent --
#
#    this proc handles events from the entry widget that we want
#    handled specially (typically, to allow navigation of the list
#    even though the focus is in the entry widget)
#
# Arguments:
#
#    w       widget pathname
#    event   a string representing the event (not necessarily an
#            actual event)
#    args    additional arguments required by particular events

proc ::combobox::HandleEvent {w event args} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options
    upvar ::combobox::${w}::oldValue oldValue

    # for all of these events, if we have a special action we'll
    # do that and do a "return -code break" to keep additional
    # bindings from firing. Otherwise we'll let the event fall
    # on through.
    switch -- $event {
    "<MouseWheel>" {
            if {[winfo ismapped $widgets(dropdown)]} {
                set D [lindex $args 0]
                # the '120' number in the following expression has
                # it's genesis in the tk bind manpage, which suggests
                # that the smallest value of %D for mousewheel events
                # will be 120. The intent is to scroll one line at a time.
                $widgets(listbox) yview scroll [expr {-($D/120)}] units
            }
        }
    "<Any-KeyPress>" {
            # if the widget is editable, clear the selection.
            # this makes it more obvious what will happen if the
            # user presses <Return> (and helps our code know what
            # to do if the user presses return)
            if {$options(-editable)} {
                $widgets(listbox) see 0
                $widgets(listbox) selection clear 0 end
                $widgets(listbox) selection anchor 0
                $widgets(listbox) activate 0
            }
        }
    "<FocusIn>" {
            set oldValue [$widgets(entry) get]
        }
    "<FocusOut>" {
            if {![winfo ismapped $widgets(dropdown)]} {
                # did the value change?
                set newValue [$widgets(entry) get]
                if {$oldValue != $newValue} {
                    CallCommand $widgets(this) $newValue
                }
            }
        }
    "<1>" {
            set editable [::combobox::GetBoolean $options(-editable)]
            if {!$editable} {
                if {[winfo ismapped $widgets(dropdown)]} {
                    $widgets(this) close
                    return -code break
                } else {
                    if {$options(-state) != "disabled"} {
                        $widgets(this) open
                        return -code break
                    }
                }
            }
        }
    "<Double-1>" {
            if {$options(-state) != "disabled"} {
                $widgets(this) toggle
                return -code break
            }
        }
    "<Tab>" {
            if {[winfo ismapped $widgets(dropdown)]} {
                ::combobox::Find $widgets(this) 0
                return -code break
            } else {
                ::combobox::SetValue $widgets(this) [$widgets(this) get]
            }
        }
    "<Escape>" {
            #            $widgets(entry) delete 0 end
            #            $widgets(entry) insert 0 $oldValue
            if {[winfo ismapped $widgets(dropdown)]} {
                $widgets(this) close
                return -code break
            }
        }
    "<Return>" {
            # did the value change?
            set newValue [$widgets(entry) get]
            if {$oldValue != $newValue} {
                CallCommand $widgets(this) $newValue
            }

            if {[winfo ismapped $widgets(dropdown)]} {
                ::combobox::Select $widgets(this) \
                  [$widgets(listbox) curselection]
                return -code break
            }

        }
    "<Next>" {
            $widgets(listbox) yview scroll 1 pages
            set index [$widgets(listbox) index @0,0]
            $widgets(listbox) see $index
            $widgets(listbox) activate $index
            $widgets(listbox) selection clear 0 end
            $widgets(listbox) selection anchor $index
            $widgets(listbox) selection set $index

        }
    "<Prior>" {
            $widgets(listbox) yview scroll -1 pages
            set index [$widgets(listbox) index @0,0]
            $widgets(listbox) activate $index
            $widgets(listbox) see $index
            $widgets(listbox) selection clear 0 end
            $widgets(listbox) selection anchor $index
            $widgets(listbox) selection set $index
        }
    "<Down>" {
            if {[winfo ismapped $widgets(dropdown)]} {
                ::combobox::tkListboxUpDown $widgets(listbox) 1
                return -code break
            } else {
                if {$options(-state) != "disabled"} {
                    $widgets(this) open
                    return -code break
                }
            }
        }
    "<Up>" {
            if {[winfo ismapped $widgets(dropdown)]} {
                ::combobox::tkListboxUpDown $widgets(listbox) -1
                return -code break
            } else {
                if {$options(-state) != "disabled"} {
                    $widgets(this) open
                    return -code break
                }
            }
        }
    }

    return ""
}

# ::combobox::DestroyHandler {w} --
#
#    Cleans up after a combobox widget is destroyed
#
# Arguments:
#
#    w    widget pathname
#
# Results:
#
#    The namespace that was created for the widget is deleted,
#    and the widget proc is removed.

proc ::combobox::DestroyHandler {w} {

    # if the widget actually being destroyed is of class Combobox,
    # crush the namespace and kill the proc. Get it? Crush. Kill.
    # Destroy. Heh. Danger Will Robinson! Oh, man! I'm so funny it
    # brings tears to my eyes.
    if {[string compare [winfo class $w] "Combobox"] == 0} {
        upvar ::combobox::${w}::widgets widgets
        upvar ::combobox::${w}::options options

        # delete the namespace and the proc which represents
        # our widget
        namespace delete ::combobox::$w
        rename $w {}
    }

    return ""
}

# ::combobox::Find
#
#    finds something in the listbox that matches the pattern in the
#    entry widget and selects it
#
#    N.B. I'm not convinced this is working the way it ought to. It
#    works, but is the behavior what is expected? I've also got a gut
#    feeling that there's a better way to do this, but I'm too lazy to
#    figure it out...
#
# Arguments:
#
#    w      widget pathname
#    exact  boolean; if true an exact match is desired
#
# Returns:
#
#    Empty string

proc ::combobox::Find {w {exact 0}} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    ## *sigh* this logic is rather gross and convoluted. Surely
    ## there is a more simple, straight-forward way to implement
    ## all this. As the saying goes, I lack the time to make it
    ## shorter...

    # use what is already in the entry widget as a pattern
    set pattern [$widgets(entry) get]

    if {[string length $pattern] == 0} {
        # clear the current selection
        $widgets(listbox) see 0
        $widgets(listbox) selection clear 0 end
        $widgets(listbox) selection anchor 0
        $widgets(listbox) activate 0
        return
    }

    # we're going to be searching this list...
    set list [$widgets(listbox) get 0 end]

    # if we are doing an exact match, try to find,
    # well, an exact match
    set exactMatch -1
    if {$exact} {
        set exactMatch [lsearch -exact $list $pattern]
    }

    # search for it. We'll try to be clever and not only
    # search for a match for what they typed, but a match for
    # something close to what they typed. We'll keep removing one
    # character at a time from the pattern until we find a match
    # of some sort.
    set index -1
    while {$index == -1 && [string length $pattern]} {
        set index [lsearch -glob $list "$pattern*"]
        if {$index == -1} {
            regsub {.$} $pattern {} pattern
        }
    }

    # this is the item that most closely matches...
    set thisItem [lindex $list $index]

    # did we find a match? If so, do some additional munging...
    if {$index != -1} {

        # we need to find the part of the first item that is
        # unique WRT the second... I know there's probably a
        # simpler way to do this...

        set nextIndex [expr {$index + 1}]
        set nextItem [lindex $list $nextIndex]

        # we don't really need to do much if the next
        # item doesn't match our pattern...
        if {[string match $pattern* $nextItem]} {
            # ok, the next item matches our pattern, too
            # now the trick is to find the first character
            # where they *don't* match...
            set marker [string length $pattern]
            while {$marker <= [string length $pattern]} {
                set a [string index $thisItem $marker]
                set b [string index $nextItem $marker]
                if {[string compare $a $b] == 0} {
                    append pattern $a
                    incr marker
                } else {
                    break
                }
            }
        } else {
            set marker [string length $pattern]
        }

    } else {
        set marker end
        set index 0
    }

    # ok, we know the pattern and what part is unique;
    # update the entry widget and listbox appropriately
    if {$exact && $exactMatch == -1} {
        # this means we didn't find an exact match
        $widgets(listbox) selection clear 0 end
        $widgets(listbox) see $index

    } elseif {!$exact} {
        # this means we found something, but it isn't an exact
        # match. If we find something that *is* an exact match we
        # don't need to do the following, since it would merely
        # be replacing the data in the entry widget with itself
        set oldstate [$widgets(entry) cget -state]
        $widgets(entry) configure -state normal
        $widgets(entry) delete 0 end
        $widgets(entry) insert end $thisItem
        $widgets(entry) selection clear
        $widgets(entry) selection range $marker end
        $widgets(listbox) activate $index
        $widgets(listbox) selection clear 0 end
        $widgets(listbox) selection anchor $index
        $widgets(listbox) selection set $index
        $widgets(listbox) see $index
        $widgets(entry) configure -state $oldstate
    }
}

# ::combobox::Select --
#
#    selects an item from the list and sets the value of the combobox
#    to that value
#
# Arguments:
#
#    w      widget pathname
#    index  listbox index of item to be selected
#
# Returns:
#
#    empty string

proc ::combobox::Select {w index} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    # the catch is because I'm sloppy -- presumably, the only time
    # an error will be caught is if there is no selection.
    if {![catch {set data [$widgets(listbox) get [lindex $index 0]]}]} {
        ::combobox::SetValue $widgets(this) $data

        $widgets(listbox) selection clear 0 end
        $widgets(listbox) selection anchor $index
        $widgets(listbox) selection set $index

    }
    $widgets(entry) selection range 0 end

    $widgets(this) close

    return ""
}

# ::combobox::HandleScrollbar --
#
#    causes the scrollbar of the dropdown list to appear or disappear
#    based on the contents of the dropdown listbox
#
# Arguments:
#
#    w       widget pathname
#    action  the action to perform on the scrollbar
#
# Returns:
#
#    an empty string

proc ::combobox::HandleScrollbar {w {action "unknown"}} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    if {$options(-height) == 0} {
        set hlimit $options(-maxheight)
    } else {
        set hlimit $options(-height)
    }

    switch -- $action {
    "grow" {
            if {$hlimit > 0 && [$widgets(listbox) size] > $hlimit} {
                pack $widgets(vsb) -side right -fill y -expand n
            }
        }
    "shrink" {
            if {$hlimit > 0 && [$widgets(listbox) size] <= $hlimit} {
                pack forget $widgets(vsb)
            }
        }
    "crop" {
            # this means the window was cropped and we definitely
            # need a scrollbar no matter what the user wants
            pack $widgets(vsb) -side right -fill y -expand n
        }
    default {
            if {$hlimit > 0 && [$widgets(listbox) size] > $hlimit} {
                pack $widgets(vsb) -side right -fill y -expand n
            } else {
                pack forget $widgets(vsb)
            }
        }
    }

    return ""
}

# ::combobox::ComputeGeometry --
#
#    computes the geometry of the dropdown list based on the size of the
#    combobox...
#
# Arguments:
#
#    w     widget pathname
#
# Returns:
#
#    the desired geometry of the listbox

proc ::combobox::ComputeGeometry {w} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    if {$options(-height) == 0 && $options(-maxheight) != "0"} {
        # if this is the case, count the items and see if
        # it exceeds our maxheight. If so, set the listbox
        # size to maxheight...
        set nitems [$widgets(listbox) size]
        if {$nitems > $options(-maxheight)} {
            # tweak the height of the listbox
            $widgets(listbox) configure -height $options(-maxheight)
        } else {
            # un-tweak the height of the listbox
            $widgets(listbox) configure -height 0
        }
        update idletasks
    }

    # compute height and width of the dropdown list
    set bd [$widgets(dropdown) cget -borderwidth]
    set height [expr {[winfo reqheight $widgets(dropdown)] + $bd + $bd}]
    if {[string length $options(-dropdownwidth)] == 0 || \
      $options(-dropdownwidth) == 0} {
        set width [winfo width $widgets(this)]
    } else {
        set m [font measure [$widgets(listbox) cget -font] "m"]
        set width [expr {$options(-dropdownwidth) * $m}]
    }

    # figure out where to place it on the screen, trying to take into
    # account we may be running under some virtual window manager
    set screenWidth [winfo screenwidth $widgets(this)]
    set screenHeight [winfo screenheight $widgets(this)]
    set rootx [winfo rootx $widgets(this)]
    set rooty [winfo rooty $widgets(this)]
    set vrootx [winfo vrootx $widgets(this)]
    set vrooty [winfo vrooty $widgets(this)]

    # the x coordinate is simply the rootx of our widget, adjusted for
    # the virtual window. We won't worry about whether the window will
    # be offscreen to the left or right -- we want the illusion that it
    # is part of the entry widget, so if part of the entry widget is off-
    # screen, so will the list. If you want to change the behavior,
    # simply change the if statement... (and be sure to update this
    # comment!)
    set x [expr {$rootx + $vrootx}]
    if {0} {
        set rightEdge [expr {$x + $width}]
        if {$rightEdge > $screenWidth} {
            set x [expr {$screenWidth - $width}]
        }
        if {$x < 0} {
            set x 0
        }
    }

    # the y coordinate is the rooty plus vrooty offset plus
    # the height of the static part of the widget plus 1 for a
    # tiny bit of visual separation...
    set y [expr {$rooty + $vrooty + [winfo reqheight $widgets(this)] + 1}]
    set bottomEdge [expr {$y + $height}]

    if {$bottomEdge >= $screenHeight} {
        # ok. Fine. Pop it up above the entry widget isntead of
        # below.
        set y [expr {($rooty - $height - 1) + $vrooty}]

        if {$y < 0} {
            # this means it extends beyond our screen. How annoying.
            # Now we'll try to be real clever and either pop it up or
            # down, depending on which way gives us the biggest list.
            # then, we'll trim the list to fit and force the use of
            # a scrollbar

            # (sadly, for windows users this measurement doesn't
            # take into consideration the height of the taskbar,
            # but don't blame me -- there isn't any way to detect
            # it or figure out its dimensions. The same probably
            # applies to any window manager with some magic windows
            # glued to the top or bottom of the screen)

            if {$rooty > [expr {$screenHeight / 2}]} {
                # we are in the lower half of the screen --
                # pop it up. Y is zero; that parts easy. The height
                # is simply the y coordinate of our widget, minus
                # a pixel for some visual separation. The y coordinate
                # will be the topof the screen.
                set y 1
                set height [expr {$rooty - 1 - $y}]

            } else {
                # we are in the upper half of the screen --
                # pop it down
                set y [expr {$rooty + $vrooty + [winfo reqheight \
                  $widgets(this)] + 1}]
                set height [expr {$screenHeight - $y}]

            }

            # force a scrollbar
            HandleScrollbar $widgets(this) crop
        }
    }

    if {$y < 0} {
        # hmmm. Bummer.
        set y 0
        set height $screenheight
    }

    set geometry [format "=%dx%d+%d+%d" $width $height $x $y]

    return $geometry
}

# ::combobox::DoInternalWidgetCommand --
#
#    perform an internal widget command, then mung any error results
#    to look like it came from our megawidget. A lot of work just to
#    give the illusion that our megawidget is an atomic widget
#
# Arguments:
#
#    w           widget pathname
#    subwidget   pathname of the subwidget
#    command     subwidget command to be executed
#    args        arguments to the command
#
# Returns:
#
#    The result of the subwidget command, or an error

proc ::combobox::DoInternalWidgetCommand {w subwidget command args} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    set subcommand $command
    set command [concat $widgets($subwidget) $command $args]
    if {[catch $command result]} {
        # replace the subwidget name with the megawidget name
        regsub $widgets($subwidget) $result $widgets(this) result

        # replace specific instances of the subwidget command
        # with our megawidget command
        switch -- $subwidget,$subcommand {
        listbox,index {
                regsub "index" $result "list index" result
            }
        listbox,insert {
                regsub "insert" $result "list insert" result
            }
        listbox,delete {
                regsub "delete" $result "list delete" result
            }
        listbox,get {
                regsub "get" $result "list get" result
            }
        listbox,size {
                regsub "size" $result "list size" result
            }
        }
        error $result

    } else {
        return $result
    }
}


# ::combobox::WidgetProc --
#
#    This gets uses as the widgetproc for an combobox widget.
#    Notice where the widget is created and you'll see that the
#    actual widget proc merely evals this proc with all of the
#    arguments intact.
#
#    Note that some widget commands are defined "inline" (ie:
#    within this proc), and some do most of their work in
#    separate procs. This is merely because sometimes it was
#    easier to do it one way or the other.
#
# Arguments:
#
#    w         widget pathname
#    command   widget subcommand
#    args      additional arguments; varies with the subcommand
#
# Results:
#
#    Performs the requested widget command

proc ::combobox::WidgetProc {w command args} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options
    upvar ::combobox::${w}::oldFocus oldFocus
    upvar ::combobox::${w}::oldFocus oldGrab

    set command [::combobox::Canonize $w command $command]

    # this is just shorthand notation...
    set doWidgetCommand [list ::combobox::DoInternalWidgetCommand \
      $widgets(this)]

    if {$command == "list"} {
        # ok, the next argument is a list command; we'll
        # rip it from args and append it to command to
        # create a unique internal command
        #
        # NB: because of the sloppy way we are doing this,
        # we'll also let the user enter our secret command
        # directly (eg: listinsert, listdelete), but we
        # won't document that fact
        set command "list-[lindex $args 0]"
        set args [lrange $args 1 end]
    }

    set result ""

    # many of these commands are just synonyms for specific
    # commands in one of the subwidgets. We'll get them out
    # of the way first, then do the custom commands.
    switch -- $command {
    bbox -
    delete -
    get -
    icursor -
    index -
    insert -
    scan -
    selection -
    xview {
            set result [eval $doWidgetCommand entry $command $args]
        }
    list-get {
            set result [eval $doWidgetCommand listbox get $args]
        }
    list-index {
            set result [eval $doWidgetCommand listbox index $args]
        }
    list-size {
            set result [eval $doWidgetCommand listbox size $args]
        }
    select {
            if {[llength $args] == 1} {
                set index [lindex $args 0]
                set result [Select $widgets(this) $index]
            } else {
                error "usage: $w select index"
            }
        }
    subwidget {
            set knownWidgets [list button entry listbox dropdown vsb]
            if {[llength $args] == 0} {
                return $knownWidgets
            }

            set name [lindex $args 0]
            if {[lsearch $knownWidgets $name] != -1} {
                set result $widgets($name)
            } else {
                error "unknown subwidget $name"
            }
        }
    curselection {
            set result [eval $doWidgetCommand listbox curselection]
        }
    list-insert {
            eval $doWidgetCommand listbox insert $args
            set result [HandleScrollbar $w "grow"]
        }
    list-delete {
            eval $doWidgetCommand listbox delete $args
            set result [HandleScrollbar $w "shrink"]
        }
    toggle {
            # ignore this command if the widget is disabled...
            if {$options(-state) == "disabled"} return

            # pops down the list if it is not, hides it
            # if it is...
            if {[winfo ismapped $widgets(dropdown)]} {
                set result [$widgets(this) close]
            } else {
                set result [$widgets(this) open]
            }
        }
    open {

            # if this is an editable combobox, the focus should
            # be set to the entry widget
            if {$options(-editable)} {
                focus $widgets(entry)
                $widgets(entry) select range 0 end
                $widgets(entry) icur end
            }

            # if we are disabled, we won't allow this to happen
            if {$options(-state) == "disabled"} {
                return 0
            }

            # if there is a -opencommand, execute it now
            if {[string length $options(-opencommand)] > 0} {
                # hmmm... should I do a catch, or just let the normal
                # error handling handle any errors? For now, the latter...
                uplevel \#0 $options(-opencommand)
            }

            # compute the geometry of the window to pop up, and set
            # it, and force the window manager to take notice
            # (even if it is not presently visible).
            #
            # this isn't strictly necessary if the window is already
            # mapped, but we'll go ahead and set the geometry here
            # since its harmless and *may* actually reset the geometry
            # to something better in some weird case.
            set geometry [::combobox::ComputeGeometry $widgets(this)]
            wm geometry $widgets(dropdown) $geometry
            update idletasks

            # if we are already open, there's nothing else to do
            if {[winfo ismapped $widgets(dropdown)]} {
                return 0
            }

            # save the widget that currently has the focus; we'll restore
            # the focus there when we're done
            set oldFocus [focus]

            # ok, tweak the visual appearance of things and
            # make the list pop up
            $widgets(button) configure -relief sunken
            raise $widgets(dropdown)
            wm deiconify $widgets(dropdown)
            tkwait visibility $widgets(dropdown)
            focus -force $widgets(dropdown)

            # force focus to the entry widget so we can handle keypress
            # events for traversal
            focus -force $widgets(entry)

            # select something by default, but only if its an
            # exact match...
            ::combobox::Find $widgets(this) 1

            # save the current grab state for the display containing
            # this widget. We'll restore it when we close the dropdown
            # list
            set status "none"
            set grab [grab current $widgets(this)]
            if {$grab != ""} {
                set status [grab status $grab]
            }
            set oldGrab [list $grab $status]
            unset grab status

            # *gasp* do a global grab!!! Mom always told me not to
            # do things like this, but sometimes a man's gotta do
            # what a man's gotta do.
            raise $widgets(dropdown)
            grab -global $widgets(this)

            # fake the listbox into thinking it has focus. This is
            # necessary to get scanning initialized properly in the
            # listbox.
            event generate $widgets(listbox) <B1-Enter>

            return 1
        }
    close {
            # if we are already closed, don't do anything...
            if {![winfo ismapped $widgets(dropdown)]} {
                return 0
            }

            # restore the focus and grab, but ignore any errors...
            # we're going to be paranoid and release the grab before
            # trying to set any other grab because we really really
            # really want to make sure the grab is released.
            catch {focus $oldFocus} result
            catch {grab release $widgets(this)}
            catch {
                set status [lindex $oldGrab 1]
                if {$status == "global"} {
                    grab -global [lindex $oldGrab 0]
                } elseif {$status == "local"} {
                    grab [lindex $oldGrab 0]
                }
                unset status
            }

            # hides the listbox
            $widgets(button) configure -relief raised
            wm withdraw $widgets(dropdown)

            # select the data in the entry widget. Not sure
            # why, other than observation seems to suggest that's
            # what windows widgets do.
            set editable [::combobox::GetBoolean $options(-editable)]
            if {$editable} {
                $widgets(entry) selection range 0 end
                $widgets(button) configure -relief raised
            }


            # magic tcl stuff (see tk.tcl in the distribution
            # lib directory)
            ::combobox::tkCancelRepeat

            return 1
        }
    cget {
            if {[llength $args] != 1} {
                error "wrong # args: should be $w cget option"
            }
            set opt [::combobox::Canonize $w option [lindex $args 0]]

            if {$opt == "-value"} {
                set result [$widgets(entry) get]
            } else {
                set result $options($opt)
            }
        }
    configure {
            set result [eval ::combobox::Configure {$w} $args]
        }
    default {
            error "bad option \"$command\""
        }
    }

    return $result
}

# ::combobox::Configure --
#
#    Implements the "configure" widget subcommand
#
# Arguments:
#
#    w      widget pathname
#    args   zero or more option/value pairs (or a single option)
#
# Results:
#
#    Performs typcial "configure" type requests on the widget

proc ::combobox::Configure {w args} {
    variable widgetOptions
    variable defaultEntryCursor

    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    if {[llength $args] == 0} {
        # hmmm. User must be wanting all configuration information
        # note that if the value of an array element is of length
        # one it is an alias, which needs to be handled slightly
        # differently
        set results {}
        foreach opt [lsort [array names widgetOptions]] {
            if {[llength $widgetOptions($opt)] == 1} {
                set alias $widgetOptions($opt)
                set optName $widgetOptions($alias)
                lappend results [list $opt $optName]
            } else {
                set optName [lindex $widgetOptions($opt) 0]
                set optClass [lindex $widgetOptions($opt) 1]
                set default [option get $w $optName $optClass]
                if {[info exists options($opt)]} {
                    lappend results [list $opt $optName $optClass $default \
                      $options($opt)]
                } else {
                    lappend results [list $opt $optName $optClass $default ""]
                }
            }
        }

        return $results
    }

    # one argument means we are looking for configuration
    # information on a single option
    if {[llength $args] == 1} {
        set opt [::combobox::Canonize $w option [lindex $args 0]]

        set optName [lindex $widgetOptions($opt) 0]
        set optClass [lindex $widgetOptions($opt) 1]
        set default [option get $w $optName $optClass]
        set results [list $opt $optName $optClass $default $options($opt)]
        return $results
    }

    # if we have an odd number of values, bail.
    if {[expr {[llength $args]%2}] == 1} {
        # hmmm. An odd number of elements in args
        error "value for \"[lindex $args end]\" missing"
    }

    # Great. An even number of options. Let's make sure they
    # are all valid before we do anything. Note that Canonize
    # will generate an error if it finds a bogus option; otherwise
    # it returns the canonical option name
    foreach {name value} $args {
        set name [::combobox::Canonize $w option $name]
        set opts($name) $value
    }

    # process all of the configuration options
    # some (actually, most) options require us to
    # do something, like change the attributes of
    # a widget or two. Here's where we do that...
    #
    # note that the handling of disabledforeground and
    # disabledbackground is a little wonky. First, we have
    # to deal with backwards compatibility (ie: tk 8.3 and below
    # didn't have such options for the entry widget), and
    # we have to deal with the fact we might want to disable
    # the entry widget but use the normal foreground/background
    # for when the combobox is not disabled, but not editable either.

    set updateVisual 0
    foreach option [array names opts] {
        set newValue $opts($option)
        if {[info exists options($option)]} {
            set oldValue $options($option)
        }

        switch -- $option {
        -background {
                set updateVisual 1
                set options($option) $newValue
            }
        -borderwidth {
                $widgets(frame) configure -borderwidth $newValue
                set options($option) $newValue
            }
        -command {
                # nothing else to do...
                set options($option) $newValue
            }
        -commandstate {
                # do some value checking...
                if {$newValue != "normal" && $newValue != "disabled"} {
                    set options($option) $oldValue
                    set message "bad state value \"$newValue\";"
                    append message " must be normal or disabled"
                    error $message
                }
                set options($option) $newValue
            }
        -cursor {
                $widgets(frame) configure -cursor $newValue
                $widgets(entry) configure -cursor $newValue
                $widgets(listbox) configure -cursor $newValue
                set options($option) $newValue
            }
        -disabledforeground {
                set updateVisual 1
                set options($option) $newValue
            }
        -disabledbackground {
                set updateVisual 1
                set options($option) $newValue
            }
        -dropdownwidth {
                set options($option) $newValue
            }
        -editable {
                set updateVisual 1
                if {$newValue} {
                    # it's editable...
                    $widgets(entry) configure -state normal \
                      -cursor $defaultEntryCursor
                } else {
                    $widgets(entry) configure -state disabled \
                      -cursor $options(-cursor)
                }
                set options($option) $newValue
            }
        -font {
                $widgets(entry) configure -font $newValue
                $widgets(listbox) configure -font $newValue
                set options($option) $newValue
            }
        -foreground {
                set updateVisual 1
                set options($option) $newValue
            }
        -height {
                $widgets(listbox) configure -height $newValue
                HandleScrollbar $w
                set options($option) $newValue
            }
        -highlightbackground {
                $widgets(frame) configure -highlightbackground $newValue
                set options($option) $newValue
            }
        -highlightcolor {
                $widgets(frame) configure -highlightcolor $newValue
                set options($option) $newValue
            }
        -highlightthickness {
                $widgets(frame) configure -highlightthickness $newValue
                set options($option) $newValue
            }
        -image {
                if {[string length $newValue] > 0} {
                    $widgets(button) configure -image $newValue
                } else {
                    $widgets(button) configure -image ::combobox::bimage
                }
                set options($option) $newValue
            }
        -maxheight {
                # ComputeGeometry may dork with the actual height
                # of the listbox, so let's undork it
                $widgets(listbox) configure -height $options(-height)
                HandleScrollbar $w
                set options($option) $newValue
            }
        -opencommand {
                # nothing else to do...
                set options($option) $newValue
            }
        -relief {
                $widgets(frame) configure -relief $newValue
                set options($option) $newValue
            }
        -selectbackground {
                $widgets(entry) configure -selectbackground $newValue
                $widgets(listbox) configure -selectbackground $newValue
                set options($option) $newValue
            }
        -selectborderwidth {
                $widgets(entry) configure -selectborderwidth $newValue
                $widgets(listbox) configure -selectborderwidth $newValue
                set options($option) $newValue
            }
        -selectforeground {
                $widgets(entry) configure -selectforeground $newValue
                $widgets(listbox) configure -selectforeground $newValue
                set options($option) $newValue
            }
        -state {
                if {$newValue == "normal"} {
                    set updateVisual 1
                    # it's enabled

                    set editable [::combobox::GetBoolean $options(-editable)]
                    if {$editable} {
                        $widgets(entry) configure -state normal
                        $widgets(entry) configure -takefocus 1
                    }

                    # note that $widgets(button) is actually a label,
                    # not a button. And being able to disable labels
                    # wasn't possible until tk 8.3. (makes me wonder
                    # why I chose to use a label, but that answer is
                    # lost to antiquity)
                    if {[info patchlevel] >= 8.3} {
                        $widgets(button) configure -state normal
                    }

                } elseif {$newValue == "disabled"} {
                    set updateVisual 1
                    # it's disabled
                    $widgets(entry) configure -state disabled
                    $widgets(entry) configure -takefocus 0
                    # note that $widgets(button) is actually a label,
                    # not a button. And being able to disable labels
                    # wasn't possible until tk 8.3. (makes me wonder
                    # why I chose to use a label, but that answer is
                    # lost to antiquity)
                    if {$::tcl_version >= 8.3} {
                        $widgets(button) configure -state disabled
                    }

                } else {
                    set options($option) $oldValue
                    set message "bad state value \"$newValue\";"
                    append message " must be normal or disabled"
                    error $message
                }

                set options($option) $newValue
            }
        -takefocus {
                $widgets(entry) configure -takefocus $newValue
                set options($option) $newValue
            }
        -textvariable {
                $widgets(entry) configure -textvariable $newValue
                set options($option) $newValue
            }
        -value {
                ::combobox::SetValue $widgets(this) $newValue
                set options($option) $newValue
            }
        -width {
                $widgets(entry) configure -width $newValue
                $widgets(listbox) configure -width $newValue
                set options($option) $newValue
            }
        -xscrollcommand {
                $widgets(entry) configure -xscrollcommand $newValue
                set options($option) $newValue
            }
        }

        if {$updateVisual} {
            UpdateVisualAttributes $w
        }
    }
}

# ::combobox::UpdateVisualAttributes --
#
# sets the visual attributes (foreground, background mostly)
# based on the current state of the widget (normal/disabled,
# editable/non-editable)
#
# why a proc for such a simple thing? Well, in addition to the
# various states of the widget, we also have to consider the
# version of tk being used -- versions from 8.4 and beyond have
# the notion of disabled foreground/background options for various
# widgets. All of the permutations can get nasty, so we encapsulate
# it all in one spot.
#
# note also that we don't handle all visual attributes here; just
# the ones that depend on the state of the widget. The rest are
# handled on a case by case basis
#
# Arguments:
#    w                widget pathname
#
# Returns:
#    empty string

proc ::combobox::UpdateVisualAttributes {w} {

    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    if {$options(-state) == "normal"} {

        set foreground $options(-foreground)
        set background $options(-background)

    } elseif {$options(-state) == "disabled"} {

        set foreground $options(-disabledforeground)
        set background $options(-disabledbackground)
    }

    $widgets(entry) configure -foreground $foreground -background $background
    $widgets(listbox) configure -foreground $foreground -background $background
    $widgets(button) configure -foreground $foreground
    $widgets(vsb) configure -background $background -troughcolor $background
    $widgets(frame) configure -background $background

    # we need to set the disabled colors in case our widget is disabled.
    # We could actually check for disabled-ness, but we also need to
    # check whether we're enabled but not editable, in which case the
    # entry widget is disabled but we still want the enabled colors. It's
    # easier just to set everything and be done with it.

    if {$::tcl_version >= 8.4} {
        $widgets(entry) configure -disabledforeground $foreground \
          -disabledbackground $background
        $widgets(button) configure -disabledforeground $foreground
        $widgets(listbox) configure -disabledforeground $foreground
    }
}

# ::combobox::SetValue --
#
#    sets the value of the combobox and calls the -command,
#    if defined
#
# Arguments:
#
#    w          widget pathname
#    newValue   the new value of the combobox
#
# Returns
#
#    Empty string

proc ::combobox::SetValue {w newValue} {

    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options
    upvar ::combobox::${w}::ignoreTrace ignoreTrace
    upvar ::combobox::${w}::oldValue oldValue

    if {[info exists options(-textvariable)] && [string length \
      $options(-textvariable)] > 0} {
        set variable ::$options(-textvariable)
        set $variable $newValue
    } else {
        set oldstate [$widgets(entry) cget -state]
        $widgets(entry) configure -state normal
        $widgets(entry) delete 0 end
        $widgets(entry) insert 0 $newValue
        $widgets(entry) configure -state $oldstate
    }

    # set our internal textvariable; this will cause any public
    # textvariable (ie: defined by the user) to be updated as
    # well
    #    set ::combobox::${w}::entryTextVariable $newValue

    # redefine our concept of the "old value". Do it before running
    # any associated command so we can be sure it happens even
    # if the command somehow fails.
    set oldValue $newValue


    # call the associated command. The proc will handle whether or
    # not to actually call it, and with what args
    CallCommand $w $newValue

    return ""
}

# ::combobox::CallCommand --
#
#   calls the associated command, if any, appending the new
#   value to the command to be called.
#
# Arguments:
#
#    w         widget pathname
#    newValue  the new value of the combobox
#
# Returns
#
#    empty string

proc ::combobox::CallCommand {w newValue} {
    upvar ::combobox::${w}::widgets widgets
    upvar ::combobox::${w}::options options

    # call the associated command, if defined and -commandstate is
    # set to "normal"
    if {$options(-commandstate) == "normal" && [string length \
      $options(-command)] > 0} {
        set args [list $widgets(this) $newValue]
        uplevel \#0 $options(-command) $args
    }
}


# ::combobox::GetBoolean --
#
#     returns the value of a (presumably) boolean string (ie: it should
#     do the right thing if the string is "yes", "no", "true", 1, etc
#
# Arguments:
#
#     value       value to be converted
#     errorValue  a default value to be returned in case of an error
#
# Returns:
#
#     a 1 or zero, or the value of errorValue if the string isn't
#     a proper boolean value

proc ::combobox::GetBoolean {value {errorValue 1}} {
    if {[catch {expr {([string trim $value]) ?1:0}} res]} {
        return $errorValue
    } else {
        return $res
    }
}

# ::combobox::convert --
#
#     public routine to convert %x, %y and %W binding substitutions.
#     Given an x, y and or %W value relative to a given widget, this
#     routine will convert the values to be relative to the combobox
#     widget. For example, it could be used in a binding like this:
#
#     bind .combobox <blah> {doSomething [::combobox::convert %W -x %x]}
#
#     Note that this procedure is *not* exported, but is intended for
#     public use. It is not exported because the name could easily
#     clash with existing commands.
#
# Arguments:
#
#     w     a widget path; typically the actual result of a %W
#           substitution in a binding. It should be either a
#           combobox widget or one of its subwidgets
#
#     args  should one or more of the following arguments or
#           pairs of arguments:
#
#           -x <x>      will convert the value <x>; typically <x> will
#                       be the result of a %x substitution
#           -y <y>      will convert the value <y>; typically <y> will
#                       be the result of a %y substitution
#           -W (or -w)  will return the name of the combobox widget
#                       which is the parent of $w
#
# Returns:
#
#     a list of the requested values. For example, a single -w will
#     result in a list of one items, the name of the combobox widget.
#     Supplying "-x 10 -y 20 -W" (in any order) will return a list of
#     three values: the converted x and y values, and the name of
#     the combobox widget.

proc ::combobox::convert {w args} {
    set result {}
    if {![winfo exists $w]} {
        error "window \"$w\" doesn't exist"
    }

    while {[llength $args] > 0} {
        set option [lindex $args 0]
        set args [lrange $args 1 end]

        switch -exact -- $option {
        -x {
                set value [lindex $args 0]
                set args [lrange $args 1 end]
                set win $w
                while {[winfo class $win] != "Combobox"} {
                    incr value [winfo x $win]
                    set win [winfo parent $win]
                    if {$win == "."} break
                }
                lappend result $value
            }
        -y {
                set value [lindex $args 0]
                set args [lrange $args 1 end]
                set win $w
                while {[winfo class $win] != "Combobox"} {
                    incr value [winfo y $win]
                    set win [winfo parent $win]
                    if {$win == "."} break
                }
                lappend result $value
            }
        -w -
        -W {
                set win $w
                while {[winfo class $win] != "Combobox"} {
                    set win [winfo parent $win]
                    if {$win == "."} break

                }
                lappend result $win
            }
        }
    }
    return $result
}

# ::combobox::Canonize --
#
#    takes a (possibly abbreviated) option or command name and either
#    returns the canonical name or an error
#
# Arguments:
#
#    w        widget pathname
#    object   type of object to canonize; must be one of "command",
#             "option", "scan command" or "list command"
#    opt      the option (or command) to be canonized
#
# Returns:
#
#    Returns either the canonical form of an option or command,
#    or raises an error if the option or command is unknown or
#    ambiguous.

proc ::combobox::Canonize {w object opt} {
    variable widgetOptions
    variable columnOptions
    variable widgetCommands
    variable listCommands
    variable scanCommands

    switch -- $object {
    command {
            if {[lsearch -exact $widgetCommands $opt] >= 0} {
                return $opt
            }

            # command names aren't stored in an array, and there
            # isn't a way to get all the matches in a list, so
            # we'll stuff the commands in a temporary array so
            # we can use [array names]
            set list $widgetCommands
            foreach element $list {
                set tmp($element) ""
            }
            set matches [array names tmp ${opt}*]
        }
    {list command} {
            if {[lsearch -exact $listCommands $opt] >= 0} {
                return $opt
            }

            # command names aren't stored in an array, and there
            # isn't a way to get all the matches in a list, so
            # we'll stuff the commands in a temporary array so
            # we can use [array names]
            set list $listCommands
            foreach element $list {
                set tmp($element) ""
            }
            set matches [array names tmp ${opt}*]
        }
    {scan command} {
            if {[lsearch -exact $scanCommands $opt] >= 0} {
                return $opt
            }

            # command names aren't stored in an array, and there
            # isn't a way to get all the matches in a list, so
            # we'll stuff the commands in a temporary array so
            # we can use [array names]
            set list $scanCommands
            foreach element $list {
                set tmp($element) ""
            }
            set matches [array names tmp ${opt}*]
        }
    option {
            if {[info exists widgetOptions($opt)] && \
              [llength $widgetOptions($opt)] == 2} {
                return $opt
            }
            set list [array names widgetOptions]
            set matches [array names widgetOptions ${opt}*]
        }
    }

    if {[llength $matches] == 0} {
        set choices [HumanizeList $list]
        error "unknown $object \"$opt\"; must be one of $choices"

    } elseif {[llength $matches] == 1} {
        set opt [lindex $matches 0]

        # deal with option aliases
        switch -- $object {
        option {
                set opt [lindex $matches 0]
                if {[llength $widgetOptions($opt)] == 1} {
                    set opt $widgetOptions($opt)
                }
            }
        }

        return $opt

    } else {
        set choices [HumanizeList $list]
        error "ambiguous $object \"$opt\"; must be one of $choices"
    }
}

# ::combobox::HumanizeList --
#
#    Returns a human-readable form of a list by separating items
#    by columns, but separating the last two elements with "or"
#    (eg: foo, bar or baz)
#
# Arguments:
#
#    list    a valid tcl list
#
# Results:
#
#    A string which as all of the elements joined with ", " or
#    the word " or "

proc ::combobox::HumanizeList {list} {

    if {[llength $list] == 1} {
        return [lindex $list 0]
    } else {
        set list [lsort $list]
        set secondToLast [expr {[llength $list] -2}]
        set most [lrange $list 0 $secondToLast]
        set last [lindex $list end]

        return "[join $most {, }] or $last"
    }
}

# This is some backwards-compatibility code to handle TIP 44
# (http://purl.org/tcl/tip/44.html). For all private tk commands
# used by this widget, we'll make duplicates of the procs in the
# combobox namespace.
#
# I'm not entirely convinced this is the right thing to do. I probably
# shouldn't even be using the private commands. Then again, maybe the
# private commands really should be public. Oh well; it works so it
# must be OK...
foreach command {TabToWindow CancelRepeat ListboxUpDown} {
    if {[llength [info commands ::combobox::tk$command]] == 1} break


    set tmp [info commands tk$command]
    set proc ::combobox::tk$command
    if {[llength [info commands tk$command]] == 1} {
        set command [namespace which [lindex $tmp 0]]
        proc $proc {args} "uplevel $command \$args"
    } else {
        if {[llength [info commands ::tk::$command]] == 1} {
            proc $proc {args} "uplevel ::tk::$command \$args"
        }
    }
}

# end of combobox.tcl


######################################################################
# icon image data.
######################################################################
image create bitmap delta48 -data {
  #define delta48_width 48
  #define delta48_height 48
  static char delta48_bits[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x80, 0x13, 0x00, 0x00,
  0x00, 0x00, 0xc0, 0x10, 0x00, 0x00, 0x00, 0x00, 0x40, 0x08, 0x00, 0x00,
  0x00, 0x00, 0x20, 0x08, 0x00, 0x00, 0x00, 0x00, 0x30, 0x0c, 0x00, 0x00,
  0x00, 0x00, 0x10, 0x04, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0e, 0x00, 0x00,
  0x00, 0x00, 0x04, 0x1b, 0x00, 0x00, 0x00, 0x00, 0x06, 0x1b, 0x00, 0x00,
  0x00, 0x00, 0x02, 0x33, 0x00, 0x00, 0x00, 0x00, 0x03, 0x2e, 0x00, 0x00,
  0x00, 0x00, 0x11, 0x6c, 0x00, 0x00, 0x00, 0x00, 0x11, 0x68, 0x00, 0x00,
  0x00, 0x80, 0x10, 0xc8, 0x00, 0x00, 0x00, 0x80, 0x10, 0xa8, 0x01, 0x00,
  0x00, 0x80, 0x08, 0x08, 0x01, 0x00, 0x00, 0x80, 0x08, 0xac, 0x03, 0x00,
  0x00, 0x80, 0x09, 0x06, 0x02, 0x00, 0x00, 0xc0, 0x09, 0xaa, 0x06, 0x00,
  0x00, 0x40, 0x09, 0x01, 0x04, 0x00, 0x00, 0xe0, 0x93, 0xae, 0x0a, 0x00,
  0x00, 0x30, 0x92, 0x06, 0x18, 0x00, 0x00, 0xb0, 0x92, 0xad, 0x1a, 0x00,
  0x00, 0x18, 0x53, 0x04, 0x30, 0x00, 0x00, 0xa8, 0x11, 0xac, 0x2a, 0x00,
  0x00, 0x0c, 0x12, 0x04, 0x60, 0x00, 0x00, 0xac, 0x12, 0xac, 0x6a, 0x00,
  0x00, 0x02, 0x14, 0x04, 0x80, 0x00, 0x00, 0xab, 0x0a, 0xae, 0xaa, 0x01,
  0x00, 0x01, 0x28, 0x02, 0x00, 0x01, 0x80, 0xab, 0x3a, 0xaf, 0xaa, 0x03,
  0x80, 0x00, 0x70, 0x0c, 0x00, 0x02, 0xc0, 0xaa, 0x5a, 0xa8, 0xaa, 0x06,
  0x40, 0x00, 0xa0, 0x08, 0x00, 0x0c, 0xa0, 0xaa, 0xea, 0xac, 0xaa, 0x0a,
  0x30, 0x00, 0x80, 0x05, 0x00, 0x18, 0xb0, 0xaa, 0xaa, 0xab, 0xaa, 0x1a,
  0x08, 0x00, 0x00, 0x04, 0x00, 0x30, 0xfc, 0xff, 0xff, 0xbe, 0xff, 0x7f,
  0xfc, 0xff, 0xff, 0xbd, 0xff, 0x7f, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  }
}

image create photo deltaGif -format gif -data {
R0lGODlhMAAwAOcAAAIyRsQWGJ4eIIYrK0aK4jZ2yXIkKOJ2hhpKgqZ+isJW
Y1IuNpBebKJifi5qumpKUsQ0OE4ySkJiip5SXoZWXnpWahZGfp5GUideptrG
yopSWtpOVnp+kiY6XiVaoIJGTphKVjxMcJ45Pso6PiFWmgYyZt5SWroeHrau
tmaClu4eJsoeHlJmhts+Q3JCSi5Kepo2PpKGkiBSlqIiIuaWllY6QrQmJmJy
kjp6zk6W9h5Ojx4yPh1Cc5YeIspaatoiJvJcaE5eekpihpR2gMIiItteZz6C
2nZqfuY2Pu9MVG5aYq4iIutGTqJOXkY+SuKKkgc6c4YyMkqO6cZCUgI2Vipi
rV4iInI2Pq5yepIiIsIrK6ErLBpKhrY+RsozM052uhIuTsJqhrJebu5UWLRM
WopWbr4iIl5mjtpCRE6G2u5eatVibqwcHjZGVoAoKr52ftoSEmo2SsYmKlaS
5pYiItZWbrpucppialKO3m5GUscqKyBGfudKUc8yMyZKeh46YiZGag42Ts46
PjZyxKcqK+JCSMpCSu7Gxh5Skp5CRtU+Px9OiUI+YropKUSC2ppOWm4eHpIp
KbV/h85qcsYuLnZOYospKoZmcqlaY5ZWWnZiZuYoLl5Scm4qLlSW8jZCXpgm
KIJCStpibAI2TrYuMK4pKp4yNtpaZr58hk6O5t1GSNE2N85CRopSciJKfuY+
Rgo2UuJOWj5+1CpmsLJkcn4uNi5QgEWG4ZI+RBpGgNU6PEZOcpAyNL5OWoBQ
ZJJKZqpSYi5eovKWnmpaflea+CZSjjFuvU2K3couMLpkcnIqLuJaXnpyhupa
Zj5Gar4uLpIuMo4uLuFLT84qKk6S7JZZZ6klJ/ZKUuJmdjZGah5WorUqKqY2
QudOUpkqK4ZabnJaegI2XiNGdspqguJGS7IaHuJeZuJaYlKS7SpGbkyK5Op0
fN5CRY5mctZESDZShtZmbhpKir4qKi5mtAIyUtpGUuBYbh5KgIJ6im5+ltIe
IlpmgnpCSuIiJgY4Tv///yH5BAEKAP8ALAAAAAAwADAAAAj+AP8JHEiwoMGD
CBMqXJgwFMOHEA0usIJpSMSLC10oMzCAFsaPBx/0yOJmwBuQKAXmccNmgBte
WFKCrFFLgIABlkRckvmRwoByloJO4PmxUxY2lgww+kI04hVILGdAiyPDGLWm
DD8M2FoOFLcLw4LJIoZVIS6c0U54AwWjQrFZUsoi1MDLUqQlS7ZYYxDiHQkc
6OQafBQtkrcA1rZ0wQQi2b0q6gQXTBTJMBxr1kpZs1FHCIlBniQLZCDCmzdr
AQiV2rbtmah8izDcCiwZk6ktW2ysWL1NHqU1N7ggchBXMjDchIiY2dZIXiNk
ayRwiecBB1m5+MhsUa2vUXM9lFb+jVuEQDhkwczIqN72Q54WLZT0CEJ1LxeC
eCQK0MZ6ptfqRv3IAx4llLByBAL2caGDbGVRw4ICpDQihwrwUYIMBJeIg8CG
uXBRVQ5YGWELLRA0oscmyCDjBSXV8MAhhzp4YAQeRKEzzyKVGKLFNJtYiEwy
QQCSzYZECjfPVTJ5IgsJ8YSggBd9IOGFF2vEkIIkkyBo34b4yZKKTARUoUMu
PHyzQR+veLGKKKjYMQI2RCJYng5V3JKSFIOQwEUuEfCjSB98rLKKIl4go0sR
RNqX4GfFfeQIBotwcQ8IpOixyhiCrqKLLoogumWiCzoCEjUOILInAmW404cX
mG7KqS7+9myp6IbCOfDlRcfI4kGkG3JyiqBJbKpICy3oYk6csk7ngSzHXCTF
LDqcmosrDUjThzRocKoIGvU88WKcc84SGUTEFMDkt9mEUY8qfCiyLTvL7APu
rPfldx1Dt2Aw5ry7iMJOEu/qQoO89M6qIIMMoWOMDNK+mAs4zSTBDhrsrLOP
OOJ8imx5Ve2XkBEeRJtoos6okQQ57MQiTDoJSJKOnDDTGqMRzSZEzTymahyn
M+e4wwTK6xQRywEvjCxngjrMM+5BqSwZj6ww27eHL7FcQw450rRQCCp7OLxx
l0gadAu0XHzrtTitAMGENFgXMoTZZis4i50GEZNn2RrrjMCzHmXwwUc30sBj
y8h50yrDIHMUlEa+vG589NG/MKGKCRzo/TiCXMR2SxoESWEMIhaELvropIf+
BzdqNMFD6ayLPlyjnuCACBQl1F57OLbnrnsHU/gAhu7A6w4FIjjULIs2UNAz
CgCjNN8888s7Lz0ATtijhD/RQy/9883TA4U2svzjCPLhUGH++einnz4sl4ji
yw6B+KP+/OF8LypRbWgiBjwKVPOJaA+5Ay4yAcACGhAiAQEAOw==
}

image create photo findImage -format gif -data {
R0lGODdhFAAUAPf/AAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAQAjUAAMIHEiwoEF3
AOQpXMiQIQB3ARC6a6fO3buHAiVWfAcPYwB1AN6pa/fQnUkAIy+qEwiy3bp07DqaPPmS3TqS
Kz/SA8ATQDyB8XoCoJczI4B2F+VBjCjvocyBCNOVS9cxAE+rUqliRHhznbunEY96dbl15kyC
Zs8OrDgzJ1uTRVnSYzcO5M8AQeu6I0oQ5DukAOAJlglPJVR5gBMifNjUqTyoAM6NK1f1auTJ
YDuuOxdTKM/NneGFHVkRLEKKE0GeFGzRdODWMhd7Xipb6FKDuAsGBAA7
}

image create photo centerDiffsImage -format gif -data {
R0lGODlhFAAUAPcAAAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAAAiUAAMIHBjAHYCD
ANwRHHjOncOHBgkRSgjRYUOEGAEYMpQRoUMA/8SJFGdwY0JyKFFSBGCuZcuSHN25bLmyo0aO
Nj+GJAkg0caNiU6q/DjToE9DQWW6rNkxUdCcBneONHhy5FCDM106zErzo82vB3XuTEm27Equ
aJd6BQsVpFSRZcmeTYuWKduM7hpW3Lv33MK/gAUGBAA7
}

image create photo firstDiffImage -format gif -data {
R0lGODlhFAAUAPcAAAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAAAiUAAMIdFevoMGD
Bd0JXBig3j9ChAxJnDixHkOBDilqlGjxIkGEIBVevHjOnbtzI1MKLAkAwEmVJN0BIKTIJUqY
AVgS+neo5kuVOv9J7Gkzpc5BFIn+XHg06SGlN1fKbDlTYiKqRRmWNFnV0FWTS7XqtGoz6six
XrMClRkxbdizbMm+jQngUKK7ao1OxTo3JliTZgUGBAA7
}

image create photo prevDiffImage -format gif -data {
R0lGODdhFAAUAPf/AAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAQAiGAAMIHCjwnDt3
5wgqLHjQHQBChgwlAtAw4cIABh9GnIjwIsOH/yIeUkTR4sWMECWW9DgQJcmOJx0SGhRR5KGR
Kxei3JjT406VMH06BECUaFCWGXsilfkP51GCKGnWdGryY9GUE4s+xfiT47mqCrsq1SmT51ao
ZYGCDevwUKK3Y8k2PLg2IAA7
}

image create photo nextDiffImage -format gif -data {
R0lGODdhFAAUAPf/AAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAQAiGAAMIHHjOncGD
5wYqVFgQACFDhhIBcJdwIUN3DgsdUjSxokWBDR9G7PixIYCTIiWeJGmx4T9ChA6x/BggJESJ
FGnWtDmSoseLGSFC3DizJMaiNE2uRLrQ5U2mQFNCJYhRak6dPHH+vGjQ4VOETasWEmrokFmO
V6OOLYt2a1iHbXWGTbswIAA7
}

image create photo lastDiffImage -format gif -data {
R0lGODlhFAAUAPcAAAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAAAiTAAMIHHjOncGD
5wYqVFgQgMOH7hIuZOgOwD9ChA4BiDiRokVDhhJtlNgxQENCIEVyLGmyIsqQI1meO5lyJEmK
BgG8VGnwZsuHOmtCvHmyEEiQh5IqiumRkNGjh5auXFgUqVSfTQtFZSrT5VWWHrmCFVhwakl3
9dKqXZvW3cR6F18enVvv7b+5eEHWXYiWrV+3AgMCADs=
}

image create photo rediffImage -format gif -data {
R0lGODdhFAAUAPf/AAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQCrPQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAQAicAAMIHEiwoMF0
7AD0euVKl8OHrhjqAgDvnDsAGDOmG2jR3TmDIAVaxFiRoMJXKF/1ypgR5UqPIWOCTIfQnc2b
ABpS/Bgg3cmUQIOqBHBxIUpYADYKLEqUp8ynUKMatFgy5LmrWEdOrDoQIcuvrnSWPJfQqFCg
YhPCAtqrrduUL8/9fIWUJs2LQ2EGmFt34MWmBNPdvKlUquEAAQEAOw==
}

image create photo markSetImage -format gif -data {
R0lGODlhFAAUAPcAAAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1Pjisd/UjtHJ
a8O4SL2qJcWqAK+SAJN6AGJiAEpKADIyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAAAiZAAMIHEhQoLqD
CAsqFAigIQB3Dd0tNKjOXSxXrmABWBABgLqCByECuAir5EYJHimKvOgqFqxXrzZ2lBhgJUaY
LV/GOpkSIqybOF3ClPlQIEShMF/lfLVzAcqPRhsKXRqTY1GCFaUy1ckTKkiRGhtapTkxa82u
ExUSJZs2qtOUbQ2ujTsQ4luvbdXNpRtA712+UeEC7ou3YEAAADt=
}

image create photo markClearImage -format gif -data {
R0lGODlhFAAUAPcAAAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1Pjisd/UjtHJ
a8O4SL2qJcWqAK+SAJN6AGJiAEpKADIyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAAAiwAAMIHEhQoLqD
CAsCWKhwIbyFANwNXBiD4UF3sVw9rLhQXQCKNTguzLgxZMePMWqo5OgqVkmVNwAIXHhDpUl3
7gCkhMkwJ02bHHfWiCkzQM5YP1cKJepRoM+kNoculEhQXc6cNW3GzNm0oFWdUSviLDgRbFST
RRsuzYpWrVaoHMsujYgVKMOPUYkCWPCQbY2iP/UuiACgr9S0NDvulQBAXd+7ZYv6bPowLdmB
By8LDAgAOw==
}

image create photo mergeChoice1Image -format gif -data {
R0lGODdhFAAUAPf/AAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAQAiIAAMIHEiwYMFz
7gAQ+meoIaGHECEeAuDuoDt35wxqFIgQAMWMGzkmVHRooseTKD1WPAgy5MCOhAZRvEizJsaR
hxrq3LkzEcWXIz+eG0qUqMujSJMixJg0AEyhRYuKVDjIUMqrMxUy5MnVkM+bAEgaOpSorNmz
X6eSnGmzZkunCT825fh2btKAADt=
}

image create photo mergeChoice2Image -format gif -data {
R0lGODdhFAAUAPf/AAAAAIAAAACAAICAAAAAgIAAgACAgMDAwMDcwKbK8P/w1P/isf/Ujv/G
a/+4SP+qJf+qANySALl6AJZiAHNKAFAyAP/j1P/Hsf+rjv+Pa/9zSP9XJf9VANxJALk9AJYx
AHMlAFAZAP/U1P+xsf+Ojv9ra/9ISP8lJf4AANwAALkAAJYAAHMAAFAAAP/U4/+xx/+Oq/9r
j/9Ic/8lV/8AVdwASbkAPZYAMXMAJVAAGf/U8P+x4v+O1P9rxv9IuP8lqv8AqtwAkrkAepYA
YnMASlAAMv/U//+x//+O//9r//9I//8l//4A/twA3LkAuZYAlnMAc1AAUPDU/+Kx/9SO/8Zr
/7hI/6ol/6oA/5IA3HoAuWIAlkoAczIAUOPU/8ex/6uO/49r/3NI/1cl/1UA/0kA3D0AuTEA
liUAcxkAUNTU/7Gx/46O/2tr/0hI/yUl/wAA/gAA3AAAuQAAlgAAcwAAUNTj/7HH/46r/2uP
/0hz/yVX/wBV/wBJ3AA9uQAxlgAlcwAZUNTw/7Hi/47U/2vG/0i4/yWq/wCq/wCS3AB6uQBi
lgBKcwAyUNT//7H//47//2v//0j//yX//wD+/gDc3AC5uQCWlgBzcwBQUNT/8LH/4o7/1Gv/
xkj/uCX/qgD/qgDckgC5egCWYgBzSgBQMtT/47H/x47/q2v/j0j/cyX/VwD/VQDcSQC5PQCW
MQBzJQBQGdT/1LH/sY7/jmv/a0j/SCX/JQD+AADcAAC5AACWAABzAABQAOP/1Mf/sav/jo//
a3P/SFf/JVX/AEncAD25ADGWACVzABlQAPD/1OL/sdT/jsb/a7j/SKr/Jar/AJLcAHq5AGKW
AEpzADJQAP//1P//sf//jv//a///SP//Jf7+ANzcALm5AJaWAHNzAFBQAPLy8ubm5tra2s7O
zsLCwra2tqqqqp6enpKSkoaGhnp6em5ubmJiYlZWVkpKSj4+PjIyMiYmJhoaGg4ODv/78KCg
pICAgP8AAAD/AP//AAAA//8A/wD//////yH5BAEAAAEALAAAAAAUABQAQAiNAAMIHEiwYEF3
AP79GzSIkMOHhAwZKkQIgLtzBguec3cxo8eNACxiHIgwpMmTIQ8dUiTSo8aRBDdynEkTIcWW
ARBGlMizJ8+VFgOcG0q0KEKWHV0qXcp0qUyYA4tKBVkxaU6UWAFMrIoR4SCfYCXe5AjgUKKz
aNMeMgT0osyaNMsihfqxpNWmQ5s2DQgAOw==
}

image create photo mergeChoice12Image -format gif -data {
R0lGODlhFAAUAPMHAAAAAAB6uQCS3CWq/0i4/47U/7Hi/////729vQAAAAAAAAAAAAAAAAAAAAAA
AAAAACH5BAEAAAgALAAAAAAUABQAAAT+ECGEECgAIYQQggghhBBCCIFiAEQIIYQQQgghhCACxRAA
AAAAAAABAAghUA4hpBRYSimllAEQAuVAQgghhBBCCCECAoRAGIQQQgghkBBCiAAIIRAGgUMIIYQQ
QggBEEQIgTAGAAAAACAAAACEEEIgDAARQgghhBBCCCGIEAIBIIQQQghBhBBCCCGEEEIIIgQKQAgh
hBBCECGEEEIImAIQggghAAAAAAAAAATEFIAQQmCUUmAppZRCCDkFIAQREIQQQgghhBBIyCkAISAI
IYRAQgghhJARAEIACiGEEEIIIQYZMACEEAAAAAAAgACAMQJACCGEEEQIIYQQAiMAhCAPQgghhBBC
CCEEQQAIIYQiADs=
}

image create photo mergeChoice21Image -format gif -data {
R0lGODlhFAAUAPMHAAAAAAB6uQCS3CWq/0i4/47U/7Hi/////729vQAAAAAAAAAAAAAAAAAAAAAA
AAAAACH5BAEAAAgALAAAAAAUABQAAAT+ECGEEEIIIYRAgQAhhBBCCCGEEEQIIWAKQAghBCAAAAAA
AACAmAIBQgiBUUoppRRYCiHkFIAQAoJAQgghhBBCCDkFAoSAIIQQQgghkBBCRgAIASGEgEIIIYQY
ZASAEEQAAAAAAAAAMOAIACGEEEIIIQQRQgiMABBCCCGIEEIIIYQQCABBhBBCCCEECkAIIoQQQggh
hBBCEBQDEEIIIYQQggghhEAxBAAAAAQAAAAAQgiUQyAhpZRSSillAAQRKIcQQgghhBBICBEAIRAG
IYRAQgghhBAiAEIIgjDIEEIIIYQQUAiAEEIgjAEAgAAAAAAAACGEEARhAIQQQgghhCAPQgghhEAA
CCEEEUIIIYQiADs=
}

image create photo nullImage

image create bitmap resize -data {
    #define resize_width 14
    #define resize_height 11
    static char resize_bits[] = {
        0x20, 0x01, 0x30, 0x03, 0x38, 0x07, 0x3c, 0x0f, 0x3e, 0x1f, 0x3f, 0x3f,
        0x3e, 0x1f, 0x3c, 0x0f, 0x38, 0x07, 0x30, 0x03, 0x20, 0x01
    }
}

# Tooltip popups

#
# tooltips version 0.1
# Paul Boyer
# Science Applications International Corp.
#

##############################
# set_tooltips gets a button's name and the tooltip string as
# arguments and creates the proper bindings for entering
# and leaving the button
##############################
proc set_tooltips {widget name} {
    global g

    bind $widget <Enter> "
    catch { after 500 { internal_tooltips_PopUp %W $name } }  g(tooltip_id)
  "
    bind $widget <Leave> "internal_tooltips_PopDown"
    bind $widget <Button-1> "internal_tooltips_PopDown"
}

##############################
# internal_tooltips_PopUp is used to activate the tooltip window
##############################
proc internal_tooltips_PopUp {wid name} {
    global g

    # get rid of other existing tooltips
    catch {destroy .tooltips_wind}

    toplevel .tooltips_wind -class ToolTip
    set size_changed 0
    set bg [option get .tooltips_wind background background]
    set fg [option get .tooltips_wind foreground foreground]

    # get the cursor position
    set X [winfo pointerx $wid]
    set Y [winfo pointery $wid]

    # add a slight offset to make tooltips fall below cursor
    set Y [expr {$Y + 20}]

    # Now pop up the new widgetLabel
    wm overrideredirect .tooltips_wind 1
    wm geometry .tooltips_wind +$X+$Y
    label .tooltips_wind.l -text $name -border 2 -relief raised \
      -background $bg -foreground $fg
    pack .tooltips_wind.l

    # make invisible
    wm withdraw .tooltips_wind
    update idletasks

    # adjust for bottom of screen
    if {($Y + [winfo reqheight .tooltips_wind]) > [winfo screenheight .]} {
        set Y [expr {$Y - [winfo reqheight .tooltips_wind] - 25}]
        set size_changed 1
    }
    # adjust for right border of screen
    if {($X + [winfo reqwidth .tooltips_wind]) > [winfo screenwidth .]} {
        set X [expr {[winfo screenwidth .] - [winfo reqwidth .tooltips_wind]}]
        set size_changed 1
    }
    # reset position
    if {$size_changed == 1} {
        wm geometry .tooltips_wind +$X+$Y
    }
    # make visible
    wm deiconify .tooltips_wind

    # make tooltip dissappear after 5 sec
    set g(tooltip_id) [after 5000 { internal_tooltips_PopDown }]
}

proc internal_tooltips_PopDown {} {
    global g

    after cancel $g(tooltip_id)
    catch {destroy .tooltips_wind}
}

# Most of this was stolen from the "CDE" package by D. J. Hagberg.
# I dig a couple more things out of the palette. -dar
proc get_cde_params {} {
    global w

    # Set defaults for all the necessary things
    set bg [option get . background background]
    set fg [option get . foreground foreground]
    set guifont [option get . buttonFontList buttonFontList]
    set txtfont [option get . FontSet FontSet]
    set listfont [option get . textFontList textFontList]
    set textbg $bg
    set textfg $fg

    # If any of these aren't set, I don't think we're in CDE after all
    if {![string length $fg]} {
        return 0
    }
    if {![string length $bg]} {
        return 0
    }
    if {![string length $guifont]} {
        return 0
    }
    if {![string length $txtfont]} {
        return 0
    }

    set guifont [string trimright $guifont ":"]
    set txtfont [string trimright $txtfont ":"]
    set listfont [string trimright $txtfont ":"]
    regsub {medium} $txtfont "bold" dlgfont

    # They don't tell us the slightly darker color they use for the
    # scrollbar backgrounds and graphics backgrounds, so we'll make
    # one up.
    set rgb_bg [winfo rgb . $bg]
    set shadow [format #%02x%02x%02x [expr {(9*[lindex $rgb_bg 0]) /2560}] \
      [expr {(9*[lindex $rgb_bg 1]) /2560}] [expr {(9*[lindex $rgb_bg 2]) \
      /2560}]]

    # If we can find the user's dt.resources file, we can find out the
    # palette and background/foreground colors
    set fh ""
    set palette ""
    set cur_rsrc ~/.dt/sessions/current/dt.resources
    set hom_rsrc ~/.dt/sessions/home/dt.resources
    if {[file readable $cur_rsrc] && [file readable $hom_rsrc]} {
        if {[file mtime $cur_rsrc] > [file mtime $hom_rsrc]} {
            if {[catch {open $cur_rsrc r} fh]} {
                set fh ""
            }
        } else {
            if {[catch {open $hom_rsrc r} fh]} {
                set fh ""
            }
        }
    } elseif {[file readable $cur_rsrc]} {
        if {[catch {open $cur_rsrc r} fh]} {
            set fh ""
        }
    } elseif {[file readable $hom_rsrc]} {
        if {[catch {open $hom_rsrc r} fh]} {
            set fh ""
        }
    }
    if {[string length $fh]} {
        set palf ""
        while {[gets $fh ln] != -1} {
            regexp "^\\*background:\[ \t]*(.*)\$" $ln nil textbg
            regexp "^\\*foreground:\[ \t]*(.*)\$" $ln nil textbg
            regexp "^\\*0\\*ColorPalette:\[ \t]*(.*)\$" $ln nil palette
            regexp "^Window.Color.Background:\[ \t]*(.*)\$" $ln nil textbg
            regexp "^Window.Color.Foreground:\[ \t]*(.*)\$" $ln nil textfg
        }
        catch {close $fh}
        #
        # If the *0*ColorPalette setting was found above, try to find the
        # indicated file in ~/.dt, $DTHOME, or /usr/dt.
        #
        if {[string length $palette]} {
            foreach dtdir {/usr/dt /etc/dt ~/.dt} {
                # This uses the last palette that we find
                if {[file readable [file join $dtdir palettes $palette]]} {
                    set palf [file join $dtdir palettes $palette]
                }
            }
            # debug-info "Using palette $palf"
            if {[string length $palf]} {
                if {![catch {open $palf r} fh]} {
                    gets $fh activetitle
                    gets $fh inactivetitle
                    gets $fh wkspc1
                    gets $fh textbg
                    gets $fh guibg ;#(*.background) - default for tk under cde
                    gets $fh menubg
                    gets $fh wkspc4
                    gets $fh iconbg ;#control panel bg too
                    close $fh

                    option add *Entry.highlightColor $activetitle userDefault
                    option add *selectColor $activetitle userDefault
                    option add *Text.highlightColor $wkspc4 userDefault
                    option add *Dialog.Background $menubg userDefault
                    option add *Menu.Background $menubg userDefault
                    option add *Menubutton.Background $menubg userDefault
                    option add *Menu.activeBackground $menubg userDefault
                    option add *Menubutton.activeBackground $menubg userDefault
                    set w(selcolor) $activetitle
                }
            }
        }
    } else {
        puts stderr "Neither ~/.dt/sessions/current/dt.resources nor"
        puts stderr "        ~/.dt/sessions/home/dt.resources was readable"
        puts stderr "   Falling back to plain X"
        return 0
    }

    #option add *Button.font $guifont userDefault
    #option add *Label.font $guifont userDefault
    #option add *Menu.font $guifont userDefault
    #option add *Menubutton.font $guifont userDefault
    #option add *Dialog.msg.font $dlgfont userDefault

    option add *Text.Background $textbg userDefault
    option add *Entry.Background $textbg userDefault
    option add *Text.Foreground $textfg userDefault
    option add *Entry.Foreground $textfg userDefault
    option add *Button.activeBackground $bg userDefault
    option add *Button.activeForeground $fg userDefault
    option add *Scrollbar.activeBackground $bg userDefault
    option add *Scrollbar.troughColor $shadow userDefault
    option add *Canvas.Background $shadow userDefault

    # These menu configs work if you use native menus.
    option add *Menu.borderWidth 1 userDefault
    option add *Menu.activeForeground $fg userDefault
    option add *Menubutton.activeForeground $fg userDefault

    # This draws a thin border around buttons
    #option add *highlightBackground $bg userDefault
    # Suppress the border
    option add *HighlightThickness 0 userDefault
    # Add it back for text and entry widgets
    option add *Text.highlightBackground $bg userDefault
    option add *Entry.highlightBackground $bg userDefault
    option add *Text.HighlightThickness 2 userDefault
    option add *Entry.HighlightThickness 1 userDefault

    return 1
}

# Maybe this could be enhanced to get configs from themes and so on?
# Right now it just sets colors so everything isn't blinding white.
proc get_aqua_params {} {
    global w

    # This doesn't seem to do anything?
    set w(selcolor) lightsteelblue

    # button highlightbackground has to be the same as background
    # or else there are little white boxes around the button "pill"
    option add *background #ebebeb userDefault
    option add *Button.highlightBackground #ebebeb userDefault

    option add *Entry.HighlightThickness 2 userDefault
    option add *Entry.highlightBackground $w(selcolor) userDefault
    #option add *Canvas.background #eeeeee userDefault
    option add *Entry.background #ffffff userDefault
    option add *Text.background white userDefault
}

###############################################################################

# run the main proc
main

