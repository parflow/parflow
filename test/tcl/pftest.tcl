set sig_digits 6

proc pftestIsEqual {a b message} {
    set pf_eps 1e-5
    if [ expr abs($a - $b) > $pf_eps ] {
	puts "FAILED : $message $a is not equal to $b"
	return 0
    } {
	return 1
    }
}

proc eps {{base 1}} {
    set eps 1e-20
    while {$base-$eps==$base} {
        set eps [expr {$eps+1e-22}]
    }
    set eps     [expr {$eps+1e-22}]
}

proc pftestFile {file message sig_digits} {
    if [file exists $file] {
	if [file exists ../correct_output/$file] {

	    set correct [pfload ../correct_output/$file]
	    set new     [pfload                $file]
	    set diff [pfmdiff $new $correct $sig_digits]
	    if {[string length $diff] != 0 } {

		set mSigDigs [lindex $diff 0]
		set maxAbsDiff [lindex $diff 1]

		set i [lindex $mSigDigs 0]
		set j [lindex $mSigDigs 1]
		set k [lindex $mSigDigs 2]

		puts "FAILED : $message"

		puts [format "\tMinimum significant digits at (% 3d, % 3d, % 3d) = %2d"\
			  $i $j $k [lindex $mSigDigs 3]]

		puts [format "\tCorrect value %e" [pfgetelt $correct $i $j $k]]
		puts [format "\tComputed value %e" [pfgetelt $new $i $j $k]]

		set elt_diff [expr abs([pfgetelt $correct $i $j $k] - [pfgetelt $new $i $j $k])]

		puts [format "\tDifference %e" $elt_diff]

		puts [format "\tMaximum absolute difference = %e" $maxAbsDiff]

		return 0
	    } {
		return 1
	    }
	} {
	    puts "FAILED : regression check output file <../correct_output/$file> does not exist"
	}
    } {
	puts "FAILED : output file <$file> not created"
	return 0
    }
}

proc pftestFileWithAbs {file message sig_digits abs_value} {
    if [file exists $file] {
	set correct [pfload ../correct_output/$file]
	set new     [pfload                $file]
	set diff [pfmdiff $new $correct $sig_digits]
	if {[string length $diff] != 0 } {

	    set mSigDigs [lindex $diff 0]
	    set maxAbsDiff [lindex $diff 1]
	    set i [lindex $mSigDigs 0]
	    set j [lindex $mSigDigs 1]
	    set k [lindex $mSigDigs 2]

	    set elt_diff [expr abs([pfgetelt $correct $i $j $k] - [pfgetelt $new $i $j $k])]

	    if [expr $elt_diff > $abs_value] {

		puts "FAILED : $message"
		puts [format "\tMinimum significant digits at (% 3d, % 3d, % 3d) = %2d"\
			  $i $j $k [lindex $mSigDigs 3]]
		puts [format "\tCorrect value %e" [pfgetelt $correct $i $j $k]]
		puts [format "\tComputed value %e" [pfgetelt $new $i $j $k]]
		puts [format "\tDifference %e" $elt_diff]

		puts [format "\tMaximum absolute difference = %e" $maxAbsDiff]
		return 0
	    }
	}

	return 1
    } {
	puts "FAILED : output file <$file> not created"
	return 0
    }
}

proc pftestParseAndEvaluateOutputForTCL {file} {

    if [file exists $file] {

	if [catch {open $file r} fileID] {
	    puts "FAILED : output file <$file> could not be read"
	} {
	    while { [gets $fileID line] >= 0} {
		if [regexp {(.*)tcl:\s*(.*)} $line match header tcl_statement] {
		    uplevel $tcl_statement
		}
	    }
	    close $fileID
	}
    } {
	puts "FAILED : output file <$file> not created"
	return 1
    }
}
