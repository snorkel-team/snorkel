var vid    = %s;
var nPages = %s;
var cid    = 0;
var pid    = 0;

// Enable tooltip functionality
$(function () {
    $('[data-toggle="tooltip"]').tooltip()
});

// Get the selector for a given candidate by number (w.r.t. to entire page group)
function candidateSelector(cid) {
    return $("span.c-" + vid + "-" + pid + "-" + cid);
}

// Cycle through candidates and highlight, by increment inc
function switchCandidate(inc) {
    var nC = parseInt($("#viewer-page-"+vid+"-"+pid).attr("data-nc"));

    // Clear highlighting and highlight new candidate
    $("span.candidate").removeClass("highlighted-candidate");
    if (cid + inc < 0) {
        cid = nC + (cid + inc);
    } else if (cid + inc > nC - 1) {
        cid = (cid + inc) - nC;
    } else {
        cid += inc;
    }
    candidateSelector(cid).addClass("highlighted-candidate");

    // Fill in caption
    $("#candidate-caption-"+vid).html($("#cdata-"+vid+"-"+pid+"-"+cid).attr("caption"));
};

$("#next-cand-" + vid).click(function() {
    switchCandidate(1);
});

$("#prev-cand-" + vid).click(function() {
    switchCandidate(-1);
});

// Switch through pages
function switchPage(inc) {
    $(".viewer-page-"+vid).hide();
    if (pid + inc < 0) {
        pid = 0;
    } else if (pid + inc > nPages - 1) {
        pid = nPages - 1;
    } else {
        pid += inc;
    }
    $("#viewer-page-"+vid+"-"+pid).show();

    // Show pagination
    $("#page-"+vid).html(pid);

    // Reset cid and set to first candidate
    cid = 0;
    switchCandidate(0);
}

$("#next-page-" + vid).click(function() {
    switchPage(1);
});

$("#prev-page-" + vid).click(function() {
    switchPage(-1);
});

// Arrow key functionality
$(document).keydown(function(e) {

    // Check that the Jupyter notebook cell the viewer is in is selected
    if ($("#viewer-"+vid).parents(".cell").hasClass("selected")) {
        switch(e.which) {
            case 74: // j
            switchCandidate(-1);
            break;

            case 73: // i
            switchPage(-1);
            break;

            case 76: // l
            switchCandidate(1);
            break;

            case 75: // k
            switchPage(1);
            break;
        }
    }
});

// Show the first page and highlight the first candidate
$("#viewer-page-"+vid+"-0").show();
switchCandidate(0);
