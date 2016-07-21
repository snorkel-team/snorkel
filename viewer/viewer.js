require.undef('viewer');

// NOTE: all elements should be selected using this.$el.find to avoid collisions with other Viewers

define('viewer', ["jupyter-js-widgets"], function(widgets) {
    var ViewerView = widgets.DOMWidgetView.extend({
        render: function() {
            this.cids   = this.model.get('cids');
            this.nPages = this.cids.length;
            this.pid = 0;
            this.cid = 0;
            this.labels = {};

            // Insert the html payload
            this.$el.append(this.model.get('html'));

            // Enable button functionality for navigation
            var that = this;
            this.$el.find("#next-cand").click(function() {
                that.switchCandidate(1);
            });
            this.$el.find("#prev-cand").click(function() {
                that.switchCandidate(-1);
            });
            this.$el.find("#next-page").click(function() {
                that.switchPage(1);
            });
            this.$el.find("#prev-page").click(function() {
                that.switchPage(-1);
            });
            this.$el.find("#label-true").click(function() {
                that.labelCandidate(true);
            });
            this.$el.find("#label-false").click(function() {
                that.labelCandidate(false);
            });

            // Arrow key functionality
            this.$el.keydown(function(e) {
                switch(e.which) {
                    case 74: // j
                    that.switchCandidate(-1);
                    break;

                    case 73: // i
                    that.switchPage(-1);
                    break;

                    case 76: // l
                    that.switchCandidate(1);
                    break;

                    case 75: // k
                    that.switchPage(1);
                    break;

                    case 84: // t
                    that.labelCandidate(true);
                    break;

                    case 70: // f
                    that.labelCandidate(false);
                    break;
                }
            });

            // Show the first page and highlight the first candidate
            this.$el.find("#viewer-page-0").show();
            this.switchCandidate(0);
        },

        // Get candidate selector for currently selected candidate, escaping id properly
        getCandidate: function() {
            return this.$el.find("."+this.cids[this.pid][this.cid].replace(/:/g, "\\:"));
        },  

        // Cycle through candidates and highlight, by increment inc
        switchCandidate: function(inc) {
            var N = this.cids[this.pid].length
            if (N == 0) { return false; }

            // Clear highlighting from previous candidate
            if (inc != 0) {
                var cPrev = this.getCandidate();
                this.setRGBABackgroundOpacity(cPrev, 0.3);
                cPrev.removeClass("highlighted-candidate");

                // Increment the cid counter
                if (this.cid + inc >= N) {
                    this.cid = N - 1;
                } else if (this.cid + inc < 0) {
                    this.cid = 0;
                } else {
                    this.cid += inc;
                }
            }

            // Highlight new candidate
            var cNext = this.getCandidate();
            this.setRGBABackgroundOpacity(cNext, 1.0);
            cNext.addClass("highlighted-candidate");

            // Push this new cid to the model
            this.model.set('selected_cid', this.cids[this.pid][this.cid]);
            this.touch();
        },

        // Switch through pages
        switchPage: function(inc) {
            this.$el.find(".viewer-page").hide();
            if (this.pid + inc < 0) {
                this.pid = 0;
            } else if (this.pid + inc > this.nPages - 1) {
                this.pid = this.nPages - 1;
            } else {
                this.pid += inc;
            }
            this.$el.find("#viewer-page-"+this.pid).show();

            // Show pagination
            this.$el.find("#page").html(this.pid);

            // Reset cid and set to first candidate
            this.cid = 0;
            this.switchCandidate(0);
        },

        // Label currently-selected candidate
        labelCandidate: function(label) {
            var c   = this.getCandidate();
            var cid = this.cids[this.pid][this.cid];
            var cl  = String(label) + "-candidate";
            var cln = String(!label) + "-candidate";

            // Flush css background-color property, so class css not overidden
            c.css("background-color", "");

            // Toggle label highlighting
            if (c.hasClass(cl)) {
                c.removeClass(cl);
                this.setRGBABackgroundOpacity(c, 1.0);
                this.labels[cid] = null;
            } else {
                c.removeClass(cln);
                c.addClass(cl);
                this.labels[cid] = label;
            }

            // Set the label and pass back to the model
            this.model.set('_labels_serialized', this.serializeDict(this.labels));
            this.touch();
        },

        // Serialization of hash maps, because traitlets Dict doesn't seem to work...
        serializeDict: function(d) {
            var s = [];
            for (var key in d) {
                s.push(key+"~~"+d[key]);
            }
            return s.join();
        },

        // Highlight spans
        setRGBABackgroundOpacity: function(el, opacity) {
            var rgx = /rgba\((\d+),\s*(\d+),\s*(\d+),\s*(\d\.\d+)\)/;
            var m   = rgx.exec(el.css("background-color"));

            // Handle rgb
            if (m == null) {
                rgx = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/;
                m   = rgx.exec(el.css("background-color"));
            }
            if (m != null) {
                el.css("background-color", "rgba("+m[1]+","+m[2]+","+m[3]+","+opacity+")");

            // TODO: Clean up this hack!!
            } else {
                el.css("background-color", "rgba(255,255,0,1)");
            }
        },
    });

    return {
        ViewerView: ViewerView
    };
});
