// To show the image //

function readURL(input, tag) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $(tag).attr('src', e.target.result);
    }

    reader.readAsDataURL(input.files[0]); // convert to base64 string
  }
}

$("#img1name").change(function () {
  readURL(this, '#img1');
});

$("#img2name").change(function () {
  readURL(this, '#img2');
});

// var img_names = ["#img1name", "#img2name"];

// // To show the name of the image//
// for (let index = 0; index < img_names.length; index++) {
//   var img_name = img_names[index];
//   // console.log(img_name, filename);
//   // console.log(1234)

//   $(img_name).bind('change', function () {
//     filename = $(img_name).val();
//     if (/^\s*$/.test(filename)) {
//       $(".file-upload").removeClass('active');
//       $("#noFile").text("No file chosen...");
//     } else {
//       $(".file-upload").addClass('active');
//       $("#noFile").text(filename);
//     }
//   });
// }