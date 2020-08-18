/*##################################################################
# Utils methods for auxiliary purposes, e.g., encoding, input/output
# and displaying
#
# @author: Angel Villar-Corrales
####################################################################*/


// converting a base64 object received by POST into a url object to be fed into
// the an image canvas or some other display
export const decodeBase64 = (encoded) =>{
    var raw = window.atob(encoded);
    var rawLength = raw.length;
    var binary_rep = new Uint8Array(new ArrayBuffer(rawLength));

    // converting each encoded char to binary
    for(var i = 0; i < rawLength; i++) {
      binary_rep[i] = raw.charCodeAt(i);
    }
    // converting into image blob and to URL object
    var blob = new Blob([binary_rep], {type: 'blob'});
    // var url = URL.createObjectURL(blob)

    return blob
}


export const lib_is_on = () =>{
  alert("Library is imported")
}

//
