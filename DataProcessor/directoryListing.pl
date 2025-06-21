
#!/usr/bin/perl
use cPanelUserConfig;

use strict;
use warnings;

use CGI;
use File::Basename;

# Get the CGI object to access the form data
my $cgi = CGI->new;

# Set the response headers to specify the content type as "application/json"
print $cgi->header(-type => "application/json");

# Get the directory path from the form data
my $directory_path = $cgi->param('directory_path');

# Open the directory and get a list of files
opendir(my $dir_handle, $directory_path) or die "Could not open directory: $!";
my @files = readdir($dir_handle);
closedir($dir_handle);

# Filter out the "." and ".." directories
@files = grep { !/^\.{1,2}$/ } @files;

# Create an array of objects representing the files
my @file_objects = ();
foreach my $filename (@files) {
  my $filepath = $directory_path . '/' . $filename;
  my $basename = basename($filename);
  my $is_directory = -d $filepath;
  push(@file_objects, {
    name => $basename,
    is_directory => $is_directory
  });
}

# Convert the array of objects to JSON format and print it
my $json_text = JSON::to_json(\@file_objects);
print $json_text;