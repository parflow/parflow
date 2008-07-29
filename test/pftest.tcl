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
	    puts "FAILED : $message $diff"
	    return 0
	} {
	    return 1
	}
    } {
	puts "FAILED : output file <$file> not created"
	return 1
    }
}

proc pftestParseAndEvaluateOutputForTCL {file} {

    if [file exists $file] {

	if [catch {open $file r} fileID] {
	    puts "FAILED : output file <$file> could not be read"
	} { 	
	    while { [gets $fileID line] >= 0} {
		if [regexp {^tcl:\s*(.*)} $line match tcl_statement] {
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

