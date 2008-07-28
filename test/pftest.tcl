set sig_digits 6

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



