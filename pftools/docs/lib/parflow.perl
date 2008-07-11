
######################################################################
#
# The perl scripts below define how `latex2html' should handle the
# commands and environments defined in `parflow.sty'.
#
# (C) 1995 Regents of the University of California.
#
# $Revision: 1.1.1.1 $
#
######################################################################

######################################################################
#
# Texinfo commands:
#
######################################################################

#---------------------------------------------------------------------
# Various verbatim-like text commands:
#
# Note: These work well as stand alone commands, but they cannot be
# used as arguments to most commands.  Exceptions are `defmac',
# `deftp', `deftypefn', `deftypvr', `index'.
#
# The following special characters are handled verbatim
#     # & ~ _ ^
#
# The following special characters are NOT handled verbatim
#     $ % \ { }
#
# These special characters may be printed by escaping with a `\'
#     $ %
#
# These special characters may also be printed by escaping with a `\'
#     # & _
#
#
# Note (HACK): To get the `~' character to always print, an empty
# HTML comment was added after the \~ command.
#
#---------------------------------------------------------------------

sub do_cmd_code{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<code>",$text,"</code>",$_);
}

sub do_cmd_file{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<code>",$text,"</code>",$_);
}

sub do_cmd_kbd{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<kbd>",$text,"</kbd>",$_);
}

sub do_cmd_key{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<code>",$text,"</code>",$_);
}

sub do_menu_key{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<code>",$text,"</code>",$_);
}

sub do_cmd_samp{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $text =~ s/~/\\~<!---->/g;                 # print `~' characters
    join('',"<samp>",$text,"</samp>",$_);
}

#---------------------------------------------------------------------
# Various non-verbatim text commands:
#---------------------------------------------------------------------

sub do_cmd_dfn{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<dfn>",$text,"</dfn>",$_);
}

sub do_cmd_var{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<var>",$text,"</var>",$_);
}

#---------------------------------------------------------------------
# Environment: defmac
#---------------------------------------------------------------------

sub do_env_defmac{
    local($_) = @_;
    local($name,$args);
    s/$next_pair_rx/$name = $&; ''/eo;
    s/^[\s]*\(.*\)/$args = $&; ''/eo;
    join('',"Macro","<P>\n",
            "<strong>",$name,"</strong>",
            "<code>",$args,"</code>","\n",$_);
}

#---------------------------------------------------------------------
# Environment: deftp
#---------------------------------------------------------------------

sub do_env_deftp{
    local($_) = @_;
    local($category,$name);
    s/$next_pair_rx/$category = $&; ''/eo;
    s/$next_pair_rx/$name = $&; ''/eo;
    join('',$category,"<P>\n",
            "<strong>",$name,"</strong>","\n",$_);
}

#---------------------------------------------------------------------
# Environment: deftypefn
#---------------------------------------------------------------------

sub do_env_deftypefn{
    local($_) = @_;
    local($class,$type,$name,$args);
    s/$next_pair_rx/$class = $&; ''/eo;
    s/$next_pair_rx/$type = $&; ''/eo;
    s/$next_pair_rx/$name = $&; ''/eo;
    s/^[\s]*\(.*\)/$args = $&; ''/eo;
    join('',$class,"<P>\n",
            "<code>",$type," </code>",
            "<strong>",$name,"</strong>",
            "<code>",$args,"</code>","\n",$_);
}

#---------------------------------------------------------------------
# Environment: deftypevr
#---------------------------------------------------------------------

sub do_env_deftypevr{
    local($_) = @_;
    local($class,$type,$name);
    s/$next_pair_rx/$class = $&; ''/eo;
    s/$next_pair_rx/$type = $&; ''/eo;
    s/$next_pair_rx/$name = $&; ''/eo;
    join('',$class,"<P>\n",
            "<code>",$type," </code>",
            "<strong>",$name,"</strong>","\n",$_);
}

######################################################################
#
# Title Page and Copyright Page definitions:
#
######################################################################

#---------------------------------------------------------------------
# Environment: TitlePage
#   Does nothing.
#---------------------------------------------------------------------

sub do_env_TitlePage{
    local($_) = @_;
    $_;
}

#---------------------------------------------------------------------
# Command: Title
#---------------------------------------------------------------------

sub do_cmd_Title{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    $TITLE = $text;
    join('',"<H1>$text</H1>\n",$_);
}

#---------------------------------------------------------------------
# Command: SubTitle
#---------------------------------------------------------------------

sub do_cmd_SubTitle{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<strong>$text</strong>\n",$_);
}

#---------------------------------------------------------------------
# Command: Author
#---------------------------------------------------------------------

sub do_cmd_Author{
    local($_) = @_;
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<strong>$text</strong>\n",$_);
}

#---------------------------------------------------------------------
# Environment: CopyrightPage
#   Does nothing.
#---------------------------------------------------------------------

sub do_env_CopyrightPage{
    local($_) = @_;
    $_;
}

######################################################################
#
# Miscellaneous commands and environments:
#
######################################################################

#---------------------------------------------------------------------
# Environment: display
#   Does nothing.
#---------------------------------------------------------------------

sub do_env_display{
    local($_) = @_;
    $_;
}

#---------------------------------------------------------------------
# Subheadings for `def' environments:
#---------------------------------------------------------------------

sub do_cmd_DESCRIPTION{
    local($_) = @_;
    join('',"\n<H4>","DESCRIPTION","</H4>\n",$_);
}

sub do_cmd_EXAMPLE{
    local($_) = @_;
    join('',"\n<H4>","EXAMPLE","</H4>\n",$_);
}

sub do_cmd_SEEALSO{
    local($_) = @_;
    join('',"\n<H4>","SEE ALSO","</H4>\n",$_);
}

sub do_cmd_NOTES{
    local($_) = @_;
    join('',"\n<H4>","NOTES","</H4>\n",$_);
}

#---------------------------------------------------------------------
# Command: vref
#   For "verbose reference", the format of this command is:
#
#     \vref{<label>}{<title>}
#
#   Prints the reference number and the <title> argument in brackets.
#---------------------------------------------------------------------

sub do_cmd_vref{
    local($_) = @_;
    local($label,$title);
    s/$next_pair_pr_rx/$label = $&; ''/eo;
    s/$next_pair_pr_rx/$title = $&; ''/eo;
    join('',&do_cmd_ref($label)," [",$title,"]",$_);
}

#---------------------------------------------------------------------
# Temporary index placeholder commands:
#---------------------------------------------------------------------

sub do_cmd_cindex{
    local($_) = @_;
    $_;
}
sub do_cmd_findex{
    &do_cmd_code(@_);
}
sub do_cmd_kindex{
    &do_cmd_code(@_);
}
sub do_cmd_pindex{
    &do_cmd_code(@_);
}
sub do_cmd_tindex{
    &do_cmd_code(@_);
}
sub do_cmd_vindex{
    &do_cmd_code(@_);
}

#---------------------------------------------------------------------
# Command: parflow
#---------------------------------------------------------------------

sub do_cmd_parflow{
    local($_) = @_;
    join('',&do_cmd_sc("ParFlow"),$_);
}

sub do_cmd_xparflow{
    local($_) = @_;
    join('',&do_cmd_sc("XParFlow"),$_);
}

sub do_cmd_pftools{
    local($_) = @_;
    join('',&do_cmd_sc("PFTools"),$_);
}

#---------------------------------------------------------------------
# Command: pfaddanchor
#---------------------------------------------------------------------

sub do_cmd_pfaddanchor{
    local($_) = @_;
    local($text);
    s/$next_pair_pr_rx/$text = $&; ''/eo;
    join('',"<a name=PFAnchor",$text,">&#160;</a>",$_);
}

######################################################################
# This next line must be the last line
######################################################################

1;

