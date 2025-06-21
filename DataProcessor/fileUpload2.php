<?php
function createDirectoryIfNotExists($path) {
    if (!is_dir($path)) {
        mkdir($path, 0755, true);
    }
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Get the form data
    $file = $_FILES['file'];
    $hashedPassword = $_POST['password'];
    $username = $_POST['username'];
    $folder = $_POST['folder'];
    
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
            echo "Incorrect password";
            exit;
        }
    } else {
        // Create the .password file with the supplied hashed password
        file_put_contents($passwordFile, $hashedPassword);
    }
    
    // Create the specified folder if it does not exist
    $targetFolder = $userDirectory . '/' . $folder;
    createDirectoryIfNotExists($targetFolder);
    
    // Move the uploaded file to the target folder
    $targetFile = $targetFolder . '/' . basename($file['name']);
    if (move_uploaded_file($file['tmp_name'], $targetFile)) {
        http_response_code(200);
        echo "File uploaded successfully";
    } else {
        http_response_code(500);
        echo "Error uploading file";
    }
} elseif ($_SERVER['REQUEST_METHOD'] === 'GET') {
    // Get the form data
    $filename = $_GET['file'];
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
            echo "Incorrect password";
            exit;
        }
    } else {
        // Create the .password file with the supplied hashed password
        file_put_contents($passwordFile, $hashedPassword);
    }
    
    // Check if the requested file exists
    $targetFolder = $userDirectory . '/' . $folder;
    $targetFile = $targetFolder . '/' . $filename;
    if (file_exists($targetFile)) {
        // Serve the file with the correct content type
        $fileInfo = finfo_open(FILEINFO_MIME_TYPE);
        $mimeType = finfo_file($fileInfo, $targetFile);
        finfo_close($fileInfo);
        error_log("file requested");
        header("Content-Type: " . $mimeType);
        header("Content-Disposition: inline; filename=\"" . basename($targetFile) . "\"");
        header("Content-Length: " . filesize($targetFile));
        readfile($targetFile);
    } else {
        http_response_code(404);
        echo "File not found";
    }
} else {
    http_response_code(405);
    echo "Invalid request method";
}
?>