# HTML.PERL by Nikos Drakos <nikos@cbl.leeds.ac.uk> 2-DEC-93
# Computer Based Learning Unit, University of Leeds.
#
# Extension to LaTeX2HTML to translate hypermedia extensions
# to LaTeX defined in html.sty to equivalent HTML commands.
#
#
#
# Modifications:
#
#  nd = Nikos Drakos <nikos@cbl.leeds.ac.uk>
#  jz = Jelle van Zeijl <jvzeijl@isou17.estec.esa.nl>

# nd 18-AUG-94 - Added do_cmd_htmladdtonavigation
# nd 26-JUL-94 - Moved do_env_latexonly from main script and added support
#                for do_cmd_latexonly
# jz 22-APR-94 - Added command htmlref
# nd 15-APR-94 - Added command htmladdnormallinkfoot
# nd  2-DEC-93 - Created

package main; 

sub do_cmd_htmladdnormallink{
    local($_) = @_;
    local($text, $url);
    s/$next_pair_pr_rx/$text = $2; ''/eo;
    s/$next_pair_pr_rx/$url = $2; ''/eo;
    join('',&make_href($url,$text),$_);
}

sub do_cmd_htmladdnormallinkfoot{
    &do_cmd_htmladdnormallink;
}

sub do_cmd_htmladdimg{
    local($_) = @_;
    local($url);
    s/$next_pair_pr_rx/$url = $2; ''/eo;
    $url = &revert_to_raw_tex($url);
    join('',&embed_image($url,"external",0,0),$_);
}

sub do_cmd_externallabels{
    local($_) = @_;
    local($URL,$labelfile);
    s/$next_pair_pr_rx/$URL = $2; ''/eo;
    s/$next_pair_pr_rx/$labelfile = $2; ''/eo;
    if (-f "$labelfile") {
	require($labelfile)}
    else {
	$warnings .= "Could not find the external label file!\n" ;
    }
    $_;
}

sub do_cmd_externalref{
    local($_) = @_;
    &process_ref($external_ref_mark,$external_ref_mark);
}

sub do_cmd_hyperref {
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $2; ''/eo;
    s/$next_pair_pr_rx//o; # Throw this away ...
    s/$next_pair_pr_rx//o; # ... and this
    &process_ref($cross_ref_mark,$cross_ref_mark,$text);
}

sub do_cmd_htmlref {
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $2; ''/eo;
    &process_ref($cross_ref_mark,$cross_ref_mark,$text);
}

# IGNORE the contents of this environment 
sub do_env_latexonly {
    "";
}

# IGNORE the argument of this command
sub do_cmd_latexonly {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $_;
}

# IGNORE the argument of this command
sub do_cmd_htmlimage {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $warnings .= "\nThe command \"htmlimage\" is only effective inside an
environment which generates an image (eg \"figure\")\n";
    $_;
}

sub do_cmd_htmladdtonavigation {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $CUSTOM_BUTTONS = $2;
    $_;
}

1;				# This must be the last line



