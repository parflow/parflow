#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

# $Id: tkdiff.tcl,v 1.1 1995/06/30 00:48:19 ssmith Exp $

###############################################################################
#                                                                             #
#                    TkDiff -- graphical diff, using Tcl/Tk                   #
#                                                                             #
# Author: John Klassa (klassa@aur.alcatel.com)                                #
# Usage:  tkdiff <file1> <file2>                                              #
#         tkdiff <file>
#         tkdiff -r<rev> <file>                                               #
#         tkdiff -r<rev1> -r<rev2> <file>                                     #
#                                                                             #
###############################################################################

###############################################################################
#
# THIS SOFTWARE IS PROVIDED BY ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN
# NO EVENT SHALL JOHN KLASSA OR ALCATEL NETWORK SYSTEMS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

global opts

###############################################################################
# Source Paul Raines' emacs bindings file if possible.
###############################################################################

set emacs_bindings_file "/homes/klassa/public/lib/tcl-tk-lib/bindings.tk"
catch {source $emacs_bindings_file}

###############################################################################
# Set defaults...
###############################################################################

set g(count)     0
set g(currdiff)  ""
set g(destroy)   ""
set g(diff)      {}
set g(pos)       0

set finfo(lbl,1) {}
set finfo(lbl,2) {}
set finfo(pth,1) {}
set finfo(pth,2) {}

set opts(diffopt) "-w"
set opts(currtag) "-background black -foreground white"
set opts(difftag) "-background white -foreground black -font 6x13bold"
set opts(textopt) "-background white -foreground black -font 6x13"
set opts(tmpdir)  "/tmp"

if {[string first "color" [winfo visual .]] >= 0} {
    set bg "#9977cc"
} else {
    set bg "black"
}

###############################################################################
# Source ~/.tkdiffrc, to override defaults (if desired).
###############################################################################

catch {source ~/.tkdiffrc}

###############################################################################
# Throw up a modal error dialog.
###############################################################################

proc do-error {msg} {
    tk_dialog .tkerror "tk error" "$msg" warning {0} "Ack!"
}

###############################################################################
# Initialize file variables.
###############################################################################

proc init-files {} {
    global argv
    global finfo
    global opts

    set finfo(lbl,1) {}
    set finfo(lbl,2) {}
    set finfo(pth,1) {}
    set finfo(pth,2) {}

    set cmd [join $argv]

    if {[regexp {^-r([^ ]+) ([^-].*)$} $cmd d r f]} {

        if {[file exists $f] != 1 || [file isdirectory $f]} {
            puts stderr "$f does not exist or is a directory."
            exit
        }

        set finfo(lbl,1) "$f (r$r)"
        set finfo(pth,1) "$opts(tmpdir)/[exec whoami][pid]-1"

        catch "exec co -p$r $f > $finfo(pth,1)"

        set finfo(lbl,2) "$f"
        set finfo(pth,2) "$f"

    } elseif {[regexp {^-r([^ ]+) -r([^ ]+) ([^-].*)$} $cmd d r1 r2 f]} {

        if {[file exists $f] != 1 || [file isdirectory $f]} {
            puts stderr "$f does not exist or is a directory."
            exit
        }

        set finfo(lbl,1) "$f (r$r1)"
        set finfo(pth,1) "$opts(tmpdir)/[exec whoami][pid]-1"

        catch "exec co -p$r1 $f > $finfo(pth,1)"

        set finfo(lbl,2) "$f (r$r2)"
        set finfo(pth,2) "$opts(tmpdir)/[exec whoami][pid]-2"

        catch "exec co -p$r2 $f > $finfo(pth,2)"

    } elseif {[regexp {^([^-].*) ([^-].*)$} $cmd d f1 f2]} {
        
        if {[file isdirectory $f1] && [file isdirectory $f2]} {
            do-error "Either <file1> or <file2> must be a plain file."
            exit
        }
        
        if {[file isdirectory $f1]} {
            set f1 "[string trimright $f1 /]/[file tail $f2]"
        } elseif {[file isdirectory $f2]} {
            set f2 "[string trimright $f2 /]/[file tail $f1]"
        }
        
        if {[file exists $f1] != 1} {
            puts stderr "$f1 does not exist."
            exit
        } elseif {[file exists $f2] != 1} {
            puts stderr "$f2 does not exist."
            exit
        }
        
        set finfo(lbl,1) "$f1"
        set finfo(pth,1) "$f1"
        set finfo(lbl,2) "$f2"
        set finfo(pth,2) "$f2"
        
    } elseif {[regexp {([^-].*)$} $cmd d f]} {

        if {[file exists $f] != 1 || [file isdirectory $f]} {
            puts stderr "$f does not exist or is a directory."
            exit
        }

        set finfo(lbl,1) "$f (last revision)"
        set finfo(pth,1) "$opts(tmpdir)/[exec whoami][pid]-1"

        catch "exec co -p $f > $finfo(pth,1)"

        set finfo(lbl,2) "$f"
        set finfo(pth,2) "$f"

    } else {
        puts stderr "usage... standard: tkdiff <file1> <file2>"
        puts stderr "         with RCS: tkdiff <file>"
        puts stderr "         with RCS: tkdiff -r<rev> <file>"
        puts stderr "         with RCS: tkdiff -r<rev1> -r<rev2> <file>"
        exit
    }
}

init-files

###############################################################################
# Set up the display...
###############################################################################

# Pack the frame that holds the text widgets and scrollbars (.f).

pack [frame .f -background black -bd 2 -relief sunken] \
        -side top -fill both -expand yes

# Pack the "old" and "new" widgets (.f.1 and .f.2).

place [frame .f.1] \
        -in .f -relx 0 -relwidth 0.49 -relheight 1.0
place [scrollbar .f.scr -command texts-yview -bd 1 -relief raised] \
        -in .f -relx 0.49 -relwidth 0.02 -relheight 1.0
place [frame .f.2 -background black] \
        -in .f -relx 0.51 -relwidth 0.49 -relheight 1.0

# Pack the text widgets and the scrollbars.

pack [label .f.1.name -textvariable finfo(lbl,1) -bg $bg -fg white] \
        -side top -fill x
pack [scrollbar .f.1.scr -command {.f.1.text yview} -bd 1 -relief raised] \
        -side left -fill y
pack [text .f.1.text -yscroll scrolls-set -width 88 -height 30] \
        -side left -expand yes -fill both
pack [label .f.2.name -textvariable finfo(lbl,2) -bg $bg -fg white] \
        -side top -fill x
pack [scrollbar .f.2.scr -command {.f.2.text yview} -bd 1 -relief raised] \
        -side right -fill y
pack [text .f.2.text -yscroll {.f.2.scr set} -width 88 -height 30 -setgrid 1] \
        -side right -expand yes -fill both

# Pack the bottom-row buttons (inside .b).

pack [frame .b -bd 2 -relief sunken] \
        -side top -fill x

# Pack the quit, help and rediff buttons.

place [button .b.quit -text quit -command {destroy .}] \
        -relx 0 -rely 0.1 -relwidth 0.08 -relheight 0.8
place [button .b.help -text help -command {do-help}] \
        -relx 0.09 -rely 0.1 -relwidth 0.08 -relheight 0.8
place [button .b.rediff -text rediff -command do-diff] \
        -relx 0.18 -rely 0.1 -relwidth 0.08 -relheight 0.8

# Pack the options widget.

place [button .b.config -text config -command customize] \
        -relx 0.27 -rely 0.1 -relwidth 0.08 -relheight 0.8

# Pack the "current diff" widgets.

place [frame .b.pos -relief raised] \
        -relx 0.36 -rely 0.1 -relwidth 0.37 -relheight 0.8
pack [label .b.pos.clabel -text "curr ("] \
        -side left -padx 1 -ipady 3
pack [menubutton .b.pos.menubutton -menu .b.pos.menubutton.menu \
        -width 5 -textvariable g(pos) -relief raised] \
        -side left -ipady 3
menu .b.pos.menubutton.menu
pack [label .b.pos.nlabel -text "of"] \
        -side left -ipady 3
pack [label .b.pos.num -textvariable g(count)] \
        -side left -ipady 3
pack [label .b.pos.phdr -text "):"] \
        -side left -ipady 3
pack [label .b.pos.curr -textvariable g(currdiff) -width 30 -relief ridge] \
        -side left -ipady 3 -expand yes -fill x

# Pack the next and prev buttons.

place [button .b.next -text next -command {move 1}] \
        -relx 0.74 -rely 0.1 -relwidth 0.08 -relheight 0.8
place [button .b.prev -text prev -command {move -1}] \
        -relx 0.83 -rely 0.1 -relwidth 0.08 -relheight 0.8
place [button .b.center -text center -command center] \
        -relx 0.92 -rely 0.1 -relwidth 0.08 -relheight 0.8

# Give the window a name & allow it to be resized.

wm title   . "TkDiff v1.0b8"
wm minsize . 1 1

# Set up text tags for the 'current diff' (the one chosen by the 'next'
# and 'prev' buttons) and any ol' diff region.  All diff regions are
# given the 'diff' tag initially...  As 'next' and 'prev' are pressed,
# to scroll through the differences, one particular diff region is
# always chosen as the 'current diff', and is set off from the others
# via the 'diff' tag -- in particular, so that it's obvious which diffs
# in the left and right-hand text widgets match.

eval ".f.1.text configure $opts(textopt)"
eval ".f.2.text configure $opts(textopt)"
eval ".f.1.text tag configure curr $opts(currtag)"
eval ".f.2.text tag configure curr $opts(currtag)"
eval ".f.1.text tag configure diff $opts(difftag)"
eval ".f.2.text tag configure diff $opts(difftag)"

###############################################################################
# Customize the display (among other things).
###############################################################################

proc customize {} {
    global opts
    global tmpopts
    global tk_version

    catch {destroy .cust}
    toplevel .cust
    wm title .cust "TkDiff Customization"
    wm minsize .cust 1 1

    set lbl(diffopt) {Options for the 'diff' process:}
    set lbl(textopt) {Text widget options (Tcl/Tk code):}
    set lbl(difftag) {Tag options for diff regions (Tcl/Tk code):}
    set lbl(currtag) {Tag options for the current diff region (Tcl/Tk code):}
    set lbl(tmpdir)  {Directory for scratch files (for the *next* session):}

    set count 0

    foreach key {diffopt textopt difftag currtag tmpdir} {
        pack [frame .cust.$count] \
                -side top -expand yes -fill both
        pack [label .cust.$count.l -text $lbl($key) -width 45 -anchor w] \
                -side left
        set tmpopts($key) $opts($key)
        pack [entry .cust.$count.e -textvariable tmpopts($key) -width 50 \
                -bd 2 -relief sunken] \
                -side left -expand yes -fill both
        if {[expr int($tk_version)] <= 3} {
            catch "bind_emacsentry .cust.$count.e"
        }
        incr count
    }

    pack [frame .cust.b] \
            -side top -expand yes -fill x
    pack [button .cust.b.apply -text apply -command apply] \
            -side left -expand yes -fill x
    pack [button .cust.b.save -text save -command save] \
            -side left -expand yes -fill x
    pack [button .cust.b.dismiss -text dismiss -command {destroy .cust}] \
            -side left -expand yes -fill x
}

###############################################################################
# Apply customization changes.
###############################################################################

proc apply {} {
    global opts
    global tmpopts

    if {[catch ".f.1.text configure $tmpopts(textopt)
                .f.2.text configure $tmpopts(textopt)
                .f.1.text tag configure curr $tmpopts(currtag)
                .f.2.text tag configure curr $tmpopts(currtag)
                .f.1.text tag configure diff $tmpopts(difftag)
                .f.2.text tag configure diff $tmpopts(difftag)"]} {
        do-error "Invalid settings!"
        eval ".f.1.text configure $opts(textopt)"
        eval ".f.2.text configure $opts(textopt)"
        eval ".f.1.text tag configure curr $opts(currtag)"
        eval ".f.2.text tag configure curr $opts(currtag)"
        eval ".f.1.text tag configure diff $opts(difftag)"
        eval ".f.2.text tag configure diff $opts(difftag)"
    } else {
        set opts(textopt) $tmpopts(textopt)
        set opts(difftag) $tmpopts(difftag)
        set opts(currtag) $tmpopts(currtag)
        set opts(diffopt) $tmpopts(diffopt)
        set opts(tmpdir)  $tmpopts(tmpdir)
    }
}

###############################################################################
# Save customization changes.
###############################################################################

proc save {} {
    global opts

    catch "exec mv [glob ~]/.tkdiffrc [glob ~]/.tkdiffrc.old"

    set fid [open ~/.tkdiffrc w]

    foreach key {diffopt textopt difftag currtag tmpdir} {
        puts $fid "set opts($key) {$opts($key)}"
    }

    close $fid
}

###############################################################################
# Scroll all windows.  Credit to Wayne Throop...
###############################################################################

proc texts-yview {args} {
    global g
    global tk_version
    
    eval .f.1.text yview $args
    
    set amt [expr int([.f.1.text index @1,1])]
    set newamt [expr $amt - int([.f.1.text index end]) + \
	                    int([.f.2.text index end])]
    
    for {set low 1; set high $g(count); set i [expr ($low+$high)/2]} \
	    {$i >= $low}                                             \
	    {set i [expr ($low+$high)/2]} {
	
	scan $g(pdiff,$i) "%s %d %d %d %d" line s1 e1 s2 e2
	
	if {$s1 > $amt} {
	    set newamt [expr $amt - $s1 + $s2]
	    set high [expr $i-1]
	} else {
	    set low [expr $i+1]
	}
    }
    
    .f.2.text yview $newamt
    
    # patch from joe@morton.rain.com (Joe Moss) -- thanks!
    if {[llength $g(diff)] > 0} {
	move [expr $i + 1] 0 0
    }
}

###############################################################################
# Set all scrollbars.  Credit to Wayne Throop...
###############################################################################

if {[expr int($tk_version)] <= 3} {
    proc scrolls-set {a1 a2 a3 a4} {
        .f.1.scr set $a1 $a2 $a3 $a4
        .f.scr   set $a1 $a2 $a3 $a4
    }
} else {
    proc scrolls-set {a1 a2} {
        .f.1.scr set $a1 $a2
        .f.scr   set $a1 $a2
    }
}

###############################################################################
# Extract the start and end lines for file1 and file2 from the diff
# stored in "line".
###############################################################################

proc extract {line} {

    if [regexp {^([0-9]+)(a|c|d)} $line d digit action] {
        set s1 $digit
        set e1 $digit
    } elseif [regexp {^([0-9]+),([0-9]+)(a|c|d)} $line d start end action] {
        set s1 $start
        set e1 $end
    }

    if [regexp {(a|c|d)([0-9]+)$} $line d action digit] {
        set s2 $digit
        set e2 $digit
    } elseif [regexp {(a|c|d)([0-9]+),([0-9]+)$} $line d action start end] {
        set s2 $start
        set e2 $end
    }

    if {[info exists s1] && [info exists s2]} {
        return "$line $s1 $e1 $s2 $e2 $action"
    } else {
        puts "Cannot parse output from diff:"
        puts "\t$line"
        exit
    }
}

###############################################################################
# Add a tag to a region.
###############################################################################

proc add-tag {wgt tag start end type new} {
    if {$type == "c" || ($type == "a" && $new) || ($type == "d" && !$new)} {
        $wgt tag add $tag $start.0 [expr $end + 1].0
    } else {
        for {set idx $start} {$idx <= $end} {incr idx} {
            $wgt tag add $tag $idx.0 $idx.6

        }
    }
}

###############################################################################
# Move the "current" diff indicator (i.e. go to the next or previous diff
# region if "relative" is 1; go to an absolute diff number if "relative"
# is 0).
###############################################################################

proc move {value {relative 1} {setpos 1}} {
    global g
    global tk_version

    scan $g(pdiff,$g(pos)) "%s %d %d %d %d %s" dummy s1 e1 s2 e2 dt

    # Replace the 'diff' tag (and remove the 'curr' tag) on the current
    # 'current' region.

    .f.1.text tag remove curr $s1.0 [expr $e1 + 1].0
    .f.2.text tag remove curr $s2.0 [expr $e2 + 1].0

    add-tag .f.1.text diff $s1 $e1 $dt 0
    add-tag .f.2.text diff $s2 $e2 $dt 1

    # Bump 'pos' (one way or the other).

    if {$relative} {
        set g(pos) [expr $g(pos) + $value]
    } else {
        set g(pos) $value
    }

    # Range check 'pos'.

    if {$g(pos) > [llength $g(diff)]} {
        set g(pos) 1
    }
    if {$g(pos) < 1} {
        set g(pos) [llength $g(diff)]
    }

    if {$g(pos) > [llength $g(diff)]} {
        set g(pos) [llength $g(diff)]
    }
    if {$g(pos) < 1} {
        set g(pos) 1
    }

    # Figure out which lines we need to address...

    scan $g(pdiff,$g(pos)) "%s %d %d %d %d %s" g(currdiff) s1 e1 s2 e2 dt

    # Remove the 'diff' tag and add the 'curr' tag to the new 'current'
    # diff region.

    .f.1.text tag remove diff $s1.0 [expr $e1 + 1].0
    .f.2.text tag remove diff $s2.0 [expr $e2 + 1].0

    add-tag .f.1.text curr $s1 $e1 $dt 0
    add-tag .f.2.text curr $s2 $e2 $dt 1

    # Move the view on both text widgets so that the new region is
    # visible.

    if {$setpos} {
        if {[expr int($tk_version)] <= 3} {
            .f.1.text yview -pickplace [expr $s1 - 1]
            .f.2.text yview -pickplace [expr $s2 - 1]
        } else {
            .f.1.text see $s1.0
            .f.2.text see $s2.0
        }
    }
}

###############################################################################
# Center the top line of the CDR in each window.
###############################################################################

proc center {} {
    global g

    scan $g(pdiff,$g(pos)) "%s %d %d %d %d %s" dummy s1 e1 s2 e2 dt

    set opix [winfo reqheight .f.1.text]
    set olin [lindex [.f.1.text configure -height] 4]
    set npix [winfo height .f.1.text]

    set h [expr $npix * $olin / ($opix * 2)]

    set o [expr $s1 - $h]
    if {$o < 0} { set o 0 }
    set n [expr $s2 - $h]
    if {$n < 0} { set n 0 }

    .f.1.text yview $o
    .f.2.text yview $n
}

###############################################################################
# Change the state on all of the diff-sensitive buttons.
###############################################################################

proc buttons {{newstate "normal"}} {
    foreach b {.b.pos.menubutton .b.next .b.prev .b.center} {
        eval "$b configure -state $newstate"
    }
}

###############################################################################
# Wipe the slate clean...
###############################################################################

proc wipe {} {
    global g
    global finfo

    set g(pos)      0
    set g(count)    0
    set g(currdiff) ""

    set finfo(lbl,1) {}
    set finfo(lbl,2) {}
    set finfo(pth,1) {}
    set finfo(pth,2) {}

    foreach mod {1 2} {
        .f.$mod.text configure -state normal
        .f.$mod.text tag remove diff 1.0 end
        .f.$mod.text tag remove curr 1.0 end
        .f.$mod.text delete 1.0 end
    }

    if {[string length $g(destroy)] > 0} {
        eval $g(destroy)
        set g(destroy) ""
    }

    .b.pos.menubutton.menu delete 0 last

    buttons disabled
}

###############################################################################
# Mark difference regions and build up the jump menu.
###############################################################################

proc mark-diffs {} {
    global g

    set different 0
    set numdiff [llength [split "$g(diff)" \n]]

    # If there are <= 30 diffs, do a one-level jump menu.  If there are
    # more than 30, do a two-level jump menu with sqrt(numdiff) in each
    # level.

    if {$numdiff <= 30} {

        set g(destroy) "$g(destroy) \
                catch \"eval .b.pos.menubutton.menu delete 0 last\"\n"

        foreach d [split "$g(diff)" \n] {

            incr g(count)

            set g(pdiff,$g(count)) "[extract $d]"

            scan $g(pdiff,$g(count)) "%s %d %d %d %d %s" dummy s1 e1 s2 e2 dt

            add-tag .f.1.text diff $s1 $e1 $dt 0
            add-tag .f.2.text diff $s2 $e2 $dt 1

            set different 1

            .b.pos.menubutton.menu add command \
                    -font 6x12 \
                    -label [format "%-6d --> %s" $g(count) $d] \
                    -command "move $g(count) 0"
        }
    } else {

        set target 0
        set increment [expr int(pow($numdiff,0.5))]

        foreach d [split "$g(diff)" \n] {

            incr g(count)

            if {$g(count) >= $target} {

                .b.pos.menubutton.menu add cascade -label $target \
                        -menu .b.pos.menubutton.menu.$target
                menu .b.pos.menubutton.menu.$target

                set current $target
                set target [expr $target + $increment]

                set g(destroy) \
                      "$g(destroy) \
                      catch \"eval .b.pos.menubutton.menu.$current \
                      delete 0 last\"\n \
                      catch \"eval destroy .b.pos.menubutton.menu.$current\"\n"
            }

            set g(pdiff,$g(count)) "[extract $d]"

            scan $g(pdiff,$g(count)) "%s %d %d %d %d %s" dummy s1 e1 s2 e2 dt

            add-tag .f.1.text diff $s1 $e1 $dt 0
            add-tag .f.2.text diff $s2 $e2 $dt 1

            set different 1

            .b.pos.menubutton.menu.$current add command \
                    -font 6x12 \
                    -label [format "%-6d --> %s" $g(count) $d] \
                    -command "move $g(count) 0"
        }
    }

    return $different
}

###############################################################################
# Compute differences (start over, basically).
###############################################################################

proc rediff {} {
    global g
    global opts
    global finfo

    wipe
    init-files

    # Read the files into their respective widgets & add line numbers.

    foreach mod {1 2} {
        set txt {}
        set hndl [open "$finfo(pth,$mod)" r]
        .f.$mod.text insert 1.0 [read $hndl]
        close $hndl
        set tgt [expr [lindex [split [.f.$mod.text index end] .] 0] - 1]
        for {set i 1} {$i <= $tgt} {incr i} {
            .f.$mod.text insert $i.0 [format "%-6d  " $i]
        }
    }

    # Diff the two files and store the summary lines into 'diff'.

    set g(diff) [exec sh -c "diff $opts(diffopt) $finfo(pth,1) $finfo(pth,2) |
                             egrep -v '^(<|>|\-)' ; exit 0"]

    # Mark up the two text widgets and go to the first diff (if there
    # is one).

    if {[mark-diffs]} {
        set g(pos) 1
        move 1 0
        buttons normal
    } else {
        buttons disabled
    }

    # Prevent tampering in the text widgets.

    foreach mod {1 2} {
        .f.$mod.text configure -state disabled
    }
}

###############################################################################
# Set the X cursor to "watch" for a window and all of its descendants.
###############################################################################

proc set-cursor {w} {
    global current

    if [string compare $w "."] {
        set current($w) [lindex [$w configure -cursor] 4]
        $w configure -cursor watch
    }
    foreach child [winfo children $w] {
        set-cursor $child
    }
}

###############################################################################
# Restore the X cursor for a window and all of its descendants.
###############################################################################

proc restore-cursor {w} {
    global current

    if [string compare $w "."] {
        catch {$w configure -cursor $current($w)}
    }
    foreach child [winfo children $w] {
        restore-cursor $child
    }
}

###############################################################################
# Flash the "rediff" button and then kick off a rediff.
###############################################################################

proc do-diff {} {
    set cur [lindex [. configure -cursor] 4]
    set-cursor .
    .b.rediff flash
    .b.rediff configure -state active
    rediff
    .b.rediff configure -state normal
    restore-cursor .
}

###############################################################################
# Throw up a help window.  Note: Couldn't get .help.f.text to do the
# equivalent of an ipadx without resorting to another level of frames...
# What gives?
###############################################################################

proc do-help {} {

    catch {destroy .help}
    toplevel .help
    wm title .help "TkDiff Help"
    wm geometry .help 70x40
    wm minsize .help 1 1

    pack [frame .help.f -background black] \
            -expand y -fill both
    pack [scrollbar .help.f.scr -command {.help.f.f.text yview}] \
            -side left -fill y -padx 1
    pack [frame .help.f.f -background white] \
            -expand y -fill both
    pack [text .help.f.f.text -wrap word -setgrid true \
            -width 55 -yscroll {.help.f.scr set} \
            -background white -foreground black] \
            -side left -expand y -fill both -padx 5
    pack [button .help.done -text done -command {destroy .help}] \
            -side bottom -fill x

    put-text .help.f.f.text {

<hdr>TkDiff</hdr>

  This tool is intended to be a graphical front-end to the standard Unix <itl>diff</itl> utility.

<hdr>Startup</hdr>

  The proper way to start <itl>TkDiff</itl> is:
<cmp>
    tkdiff file1 file2 &</cmp>
or<cmp>
    tkdiff file &</cmp>
or<cmp>
    tkdiff -rREV file &</cmp>
or<cmp>
    tkdiff -rREV1 -rREV2 file &</cmp>

  In the first form, one or the other (or both) of the arguments must be the name of a plain old text file.  Symbolic links (and other such magic) are acceptable, but at least one or the other (or both) of the filename arguments must point to a real file rather than to a directory.  In the last two forms, <cmp>REV</cmp> (or <cmp>REV1</cmp> and <cmp>REV2</cmp>) must be a valid RCS version number for <cmp>file</cmp>.  In the second form, <cmp>file</cmp> is compared with the the version most recently checked into RCS (i.e. it's the equivalent of <cmp>rcsdiff file</cmp>).

<hdr>Layout</hdr>

  The left-most text widget displays the contents of <cmp>file1</cmp>, the most recently checked-in version, <cmp>REV</cmp> or <cmp>REV1</cmp>, respectively (as per the startup options described above); the right-most widget displays the contents of <cmp>file2</cmp>, <cmp>file</cmp> or <cmp>REV2</cmp>, respectively.

  All difference regions (DRs) are automatically highlighted in <bld>bold-face</bld> type.  The <itl>current</itl> DR (or CDR) is highlighted in <rev>reverse</rev> video, to set it apart from the other DRs.

  The CDR on the left matches the one on the right.  The CDR can be moved by means of the <btn>.next.</btn> and <btn>.prev.</btn> buttons, as well as by the menu under the <cmp>curr (<btn>.X.</btn> of Y)</cmp> button on the bottom of the screen.  The "meta" scrollbar (below) likewise moves the CDR.

  Line numbers are automatically prepended to the lines in both widgets.

<hdr>Operations</hdr>

  <btn> .quit. </btn>:  Terminates <itl>TkDiff</itl>.

  <btn> .help. </btn>:  Generates this information.

  <btn>.rediff.</btn>:  Recomputes the differences between the two files whose names appear at the top of the <itl>TkDiff</itl> window.

  <btn>.config.</btn>:  Pops up a dialog box from which display (and other) options can be changed and saved.

  The label next to the <cmp>curr (<btn>.X.</btn> of Y)</cmp> area shows the <itl>diff</itl> mnemonic for the CDR.  The <cmp>curr (<btn>.X.</btn> of Y)</cmp> button itself allows you to select a DR to become the CDR.  This allows you to jump to any DR without having to traverse the intervening list one step at a time.

  <btn> .next. </btn>:  Takes you to the "next" DR.

  <btn> .prev. </btn>:  Takes you to the "previous" DR.

  <btn>.center.</btn>:  Centers the CDRs in their respective widgets.

<hdr>Scrolling</hdr>

  The left and right text widgets can be scrolled independently via the left-most and right-most scrollbars, respectively.  The middle scrollbar is a "meta" scrollbar, and scrolls both text widgets in a synchronized fashion.

<hdr>Credits</hdr>

  Thanks go to Wayne Throop for beta testing and for giving valuable suggestions along the way.  Wayne also came up with the synchronized scrolling mechanism... Additional credit goes to John Heidemann, author of <itl>Klondike</itl> (a great Tk-based Solitaire game).  I shamelessly stole John's window tags routines out of <itl>Klondike</itl> and used them here.

  Countless others have offered their suggestions and encouragement.  I thank you one and all!

<hdr>Comments</hdr>

  Questions and comments should be sent to John Klassa at <itl>klassa@aur.alcatel.com</itl>.

<hdr>Disclaimer</hdr>

  <bld>THIS SOFTWARE IS PROVIDED BY ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL JOHN KLASSA OR ALCATEL NETWORK SYSTEMS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.</bld>

    }

    .help.f.f.text configure -state disabled
}

###############################################################################
# Get things going...
###############################################################################

wipe

#update
#toplevel .message
#wm title .message "TkDiff Notice"
#pack [label .message.msg -foreground black -background white \
#        -width 30 -text "Computing differences..."] \
#        -side left -fill x -expand yes -ipady 20
#update

# Size up the window...

.f configure -width \
        [expr [winfo reqwidth .f.scr] + \
             ([winfo reqwidth .f.1.text] + [winfo reqwidth .f.1.scr])* 2]
.f configure -height \
        [expr [winfo reqheight .f.2.text] + [winfo reqheight .f.2.name]]
.b configure -width [winfo reqwidth .f]
.b configure -height [expr int([winfo reqheight .b.quit] * 1.5)]

update idletasks

# Compute the differences...

do-diff

#destroy .message

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

    $tw configure -font -*-Times-Medium-R-Normal-*-140-*

    $tw tag configure bld -font -*-Times-Bold-R-Normal-*-140-*
    $tw tag configure cmp -font -*-Courier-Bold-R-Normal-*-120-*
    $tw tag configure hdr -font -*-Times-Bold-R-Normal-*-180-* -underline 1
    $tw tag configure itl -font -*-Times-Medium-I-Normal-*-140-*
    $tw tag configure rev -foreground white -background black

    $tw tag configure btn \
            -font -*-Courier-Medium-R-Normal-*-120-* \
            -foreground black -background white \
            -relief groove -borderwidth 2
            
    $tw mark set insert 0.0

    set t $txt

    while {[regexp -indices {<([^@>]*)>} $t match inds] == 1} {

        set start [lindex $inds 0]
        set end [lindex $inds 1]
        set keyword [string range $t $start $end]

        set oldend [$tw index end]

        $tw insert end [string range $t 0 [expr $start - 2]]

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
        
        set t [string range $t [expr $end + 2] end]
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
