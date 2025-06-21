<?php
header("Content-Type: application/json");

function createDirectoryIfNotExists($path) {
    if (!is_dir($path)) {
        mkdir($path, 0755, true);
    }
}

// Get the form data
$hashedPassword = $_GET['password'];
$username = $_GET['username'];
$folder = $_GET['folder'];

// Create the user directory with the md5-256 hashed username
$userDirectory = 'users/' . hash('sha256', $username);
createDirectoryIfNotExists($userDirectory);

// Check for the .password file
$passwordFile = $userDirectory . '/.password';
if (file_exists($passwordFile)) {
    $storedPassword = file_get_contents($passwordFile);
    
    // If the stored password does not match the supplied password, return failure
    if ($storedPassword !== $hashedPassword) {
        http_response_code(403);
        echo json_encode(['error' => 'Incorrect password']);
        exit;
    }
} else {
    // Create the .password file with the supplied hashed password
    file_put_contents($passwordFile, $hashedPassword);
}

// Set the directory path based on the folder parameter
$directory_path = empty($folder) ? $userDirectory : $userDirectory . '/' . $folder;

// Open the directory and get a list of files
$dir_handle = opendir($directory_path);
if (!$dir_handle) {
    die("Could not open directory: " . $directory_path);
}

$files = array();
while (($filename = readdir($dir_handle)) !== false) {
    // Filter out the "." and ".." directories and hidden files (starting with ".")
    if ($filename !== "." && $filename !== ".." && substr($filename, 0, 1) !== '.') {
        $filepath = $directory_path . '/' . $filename;
        $is_directory = is_dir($filepath);
        $files[] = array(
            'name' => $filename,
            'is_directory' => $is_directory
        );
    }
}
closedir($dir_handle);

// Convert the array of objects to JSON format and print it
$json_text = json_encode($files);
echo $json_text;
?>