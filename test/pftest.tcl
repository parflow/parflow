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

proc pftestFile {file message sig_digits} {
    if [file exists $file] {
	set correct [pfload correct_output/$file]
	set new     [pfload                $file]
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

	    puts [format "\tMaximum absolute difference = %e" $maxAbsDiff]
	    
	    puts [format "\tCorrect value %e" [pfgetelt $correct $i $j $k]]
	    puts [format "\tComputed value %e" [pfgetelt $new $i $j $k]]

	    return 0
	} {
	    return 1
	}
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

