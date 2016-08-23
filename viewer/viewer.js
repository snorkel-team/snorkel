require.undef('viewer');

// NOTE: all elements should be selected using this.$el.find to avoid collisions with other Viewers

define('viewer', ["jupyter-js-widgets"], function(widgets) {
    var ViewerView = widgets.DOMWidgetView.extend({
        render: function() {
            this.cids   = this.model.get('cids');
            this.nPages = this.cids.length;
            this.pid = 0;
            this.cid = 0;

            // Insert the html payload
            this.$el.append(this.model.get('html'));

            // Initialize all labels from previous sessions
            this.labels = this.deserializeDict(this.model.get('_labels_serialized'));
            for (var i=0; i < this.nPages; i++) {
                this.pid = i;
                for (var j=0; j < this.cids[i].length; j++) {
                    this.cid = j;
                    if (this.cids[i][j] in this.labels) {
                        this.markCurrentCandidate(false);
                    }
                }
            }
            this.pid = 0;
            this.cid = 0;

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
            return this.$el.find("."+this.cids[this.pid][this.cid]);
        },  

        // Color the candidate correctly according to registered label, as well as set highlighting
        markCurrentCandidate: function(highlight) {
            var cid  = this.cids[this.pid][this.cid];
            var tags = this.$el.find("."+cid);

            // Clear color classes
            tags.removeClass("candidate-h");
            tags.removeClass("true-candidate");
            tags.removeClass("true-candidate-h");
            tags.removeClass("false-candidate");
            tags.removeClass("false-candidate-h");
            tags.removeClass("highlighted");

            if (highlight) {
                if (cid in this.labels) {
                    tags.addClass(String(this.labels[cid]) + "-candidate-h");
                } else {
                    tags.addClass("candidate-h");
                }
            
            // If un-highlighting, leave with first non-null coloring
            } else {
                var that = this;
                tags.each(function() {
                    var cids = $(this).attr('class').split(/\s+/).map(function(item) {
                        return parseInt(item);
                    });
                    cids.sort();
                    console.log(cids);
                    for (var i in cids) {
                        if (cids[i] in that.labels) {
                            var label = that.labels[cids[i]];
                            $(this).addClass(String(label) + "-candidate");
                            $(this).removeClass(String(!label) + "-candidate");
                            break;
                        }
                    }
                });
            }

            // Extra highlighting css
            if (highlight) {
                tags.addClass("highlighted");
            }
        },

        // Cycle through candidates and highlight, by increment inc
        switchCandidate: function(inc) {
            var N = this.cids[this.pid].length
            if (N == 0) { return false; }

            // Clear highlighting from previous candidate
            if (inc != 0) {
                this.markCurrentCandidate(false);

                // Increment the cid counter
                if (this.cid + inc >= N) {
                    this.cid = N - 1;
                } else if (this.cid + inc < 0) {
                    this.cid = 0;
                } else {
                    this.cid += inc;
                }
            }
            this.markCurrentCandidate(true);

            // Push this new cid to the model
            this.model.set('_selected_cid', this.cids[this.pid][this.cid]);
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
                this.send({event: 'delete_label', cid: cid});
            } else {
                c.removeClass(cln);
                c.addClass(cl);
                this.labels[cid] = label;
                this.send({event: 'set_label', cid: cid, value: label});
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

        // Deserialization of hash maps
        deserializeDict: function(s) {
            var d = {};
            var entries = s.split(/,/);
            var kv;
            for (var i in entries) {
                kv = entries[i].split(/~~/);
                if (kv[1] == "true") {
                    d[kv[0]] = true;
                } else if (kv[1] == "false") {
                    d[kv[0]] = false;
                }
            }
            return d;
        },
    });

    return {
        ViewerView: ViewerView
    };
});
