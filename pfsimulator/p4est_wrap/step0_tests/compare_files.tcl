set sig_digits 6

proc pftestIsEqual {a b message} {
    set pf_eps 1e-5
    if [ expr abs($a - $b) > $pf_eps ]  {
	puts "FAILED : $message $a is not equal to $b"
	return 0
    } else {
	return 1
    }
}

proc pftestFile {file1 file2 message sig_digits} {
    if { [ file exists $file1 ] && [ file exists $file2 ] } {
	set correct [pfload $file1]
	set new     [pfload $file2]
	set diff [pfmdiff $new $correct $sig_digits]
	if {[string length $diff] != 0 } {
	    puts "FAILED : $message"

	    set mSigDigs [lindex $diff 0]
	    set maxAbsDiff [lindex $diff 1]

	    set i [lindex $mSigDigs 0]
	    set j [lindex $mSigDigs 1]
	    set k [lindex $mSigDigs 2] 
	    puts [format "\tMinimum significant digits at (% 3d, % 3d, % 3d) = %2d"\
		      $i $j $k [lindex $mSigDigs 3]]

            puts [format "\tComputed value with standard Parflow %e" [pfgetelt $correct $i $j $k]]
	    puts [format "\tComputed value with Parflow + p4est %e" [pfgetelt $new $i $j $k]]

	    set elt_diff [expr abs([pfgetelt $correct $i $j $k] - [pfgetelt $new $i $j $k])]

	    puts [format "\tDifference %e" $elt_diff]

	    puts [format "\tMaximum absolute difference = %e" $maxAbsDiff]

	    return 0
        } else {
	    return 1
	}
    } else {
        puts "FAILED : output file <$file1> or <$file2> not created"
	return 0
    }
}
