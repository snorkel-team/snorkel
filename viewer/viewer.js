var vid    = %s;
var nPages = %s;
var cand   = 0;
var page   = 0;

// Get the selector for a given candidate by number (w.r.t. to entire page group)
function cid(n) {
    return "span.c-" + n + "-" + vid;
}

// Switch through pages
function switchPage(inc) {
    $(".viewer-page-"+vid).hide();
    if (page + inc < 0) {
        page = 0;
    } else if (page + inc > nPages) {
        page = nPages - 1;
    } else {
        page += inc;
    }
    $("#viewer-page-"+vid+"-"+page).show();
}

$("#next-page-" + vid).click(function() {
    switchPage(1);
});

$("#prev-page-" + vid).click(function() {
    switchPage(-1);
});

// Get the total number of candidates on the page
// TODO: Obsolete!
function getNumberOfCandidates() {
    var nC = 0;
    $("li.context-li:visible").each(function() {
        nC += parseInt($(this).attr("data-nc"));
    });
    return nC;
}

// Cycle through candidates and highlight, by increment inc
function switchCandidate(inc) {
    var nC = parseInt($("#viewer-page-"+vid+"-"+page).attr("data-nc"));

    // Clear highlighting and highlight new candidate
    $("span.candidate").removeClass("highlighted-candidate");
    if (cand + inc < 0) {
        cand = nC + (cand + inc);
    } else if (cand + inc > nC - 1) {
        cand = (cand + inc) - nC;
    } else {
        cand += inc;
    }
    var c = $(cid(cand));
    c.addClass("highlighted-candidate");

    // Fill in caption
    $("#candidate-caption-"+vid).html($("#cdata-"+cand+"-"+vid).attr("caption"));
};

$("#next-cand-" + vid).click(function() {
    switchCandidate(1);
});

$("#prev-cand-" + vid).click(function() {
    switchCandidate(-1);
});

// Arrow key functionality
$(document).keydown(function(e) {

    // Check that the Jupyter notebook cell the viewer is in is selected
    if ($("#viewer-"+vid).parents(".cell").hasClass("selected")) {
        switch(e.which) {
            case 37: // Left
            switchCandidate(-1);
            break;

            case 38: // Up
            switchPage(-1);
            break;

            case 39: // Right
            switchCandidate(1);
            break;

            case 40: // Down
            switchPage(1);
            break;
        }
    }
});

// Show the first page and highlight the first candidate
$("#viewer-page-"+vid+"-0").show();
switchCandidate(0);
