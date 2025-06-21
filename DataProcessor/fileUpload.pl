#!/usr/bin/perl
use cPanelUserConfig;

use strict;
use warnings;

use CGI;
use Crypt::CBC;

# Get the CGI object to access the form data
my $cgi = CGI->new;

# Get the encrypted ID string from the form data
my $encrypted_id = $cgi->param("id");

# Decrypt the ID string using a secret key
my $key = "mysecretkey";
my $cipher = Crypt::CBC->new(
    -key    => $key,
    -cipher => 'Blowfish'
);
my $id_string = $cipher->decrypt_hex($encrypted_id);

# Get the uploaded file data
my $file_data = $cgi->upload("file");

# If the file data is defined, save the file to the server
if ($file_data) {
  my $filename = $cgi->param("file");
  my $upload_path = "/home4/oktqajmy/public_html/bike-dogbite-me/$id_string/$filename";

  open(my $fh, '>', $upload_path) or die "Could not open file '$upload_path' $!";
  while (my $chunk = $file_data->getline) {
    print $fh $chunk;
  }
  close($fh);

  # Print a success message
  print $cgi->header(-type => "text/plain");
  print "File uploaded successfully to $upload_path\n";
} else {
  # Print an error message if no file data was received
  print $cgi->header(-type => "text/plain", -status => "400 Bad Request");
  print "No file data received\n";
}