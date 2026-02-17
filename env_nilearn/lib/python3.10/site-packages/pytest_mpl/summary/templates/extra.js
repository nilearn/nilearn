// Enable tooltips
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl)
})

// Search, sort and filter
var options = {
    valueNames: ['collected-sort', 'test-name', 'status-sort', 'rms-sort', 'filter-classes',
        'rms-value', 'baseline-hash-value', 'result-hash-value']
};
var resultsList = new List('results', options);

var filterClasses = [];
var filterElements = document.getElementById('filterForm').getElementsByClassName('filter');
for (var i = 0, elem; elem = filterElements[i++];) {
    filterClasses.push(elem.id);
}
countClasses();

// Get and apply initial search parameters from URL
var searchParams = new URLSearchParams(window.location.search);
if (window.location.search.length > 0) {
    applyURL();
} else {  // If no parameters, apply default but don't update the URL
    resultsList.sort('status-sort', {order: "desc"});
}
// Show page after initial filtering to prevent flashing
document.getElementById('resultslist').style.display = null;

// Show a message if no tests match current filters
var alertPlaceholder = document.getElementById('noResultsAlert');
warnIfNone();  // Initialize
resultsList.on('updated', function () {
    warnIfNone();
})

// Record URL parameters after new sort (but do not update URL yet)
resultsList.on('sortComplete', function updateSortURL() {
    var sortElements = document.getElementsByClassName('sort');
    for (var i = 0, elem; elem = sortElements[i++];) {
        if (elem.checked) {
            searchParams.set('sort', elem.dataset['sort']);
            searchParams.set('order', getSortOrder(elem));
            break;
        }
    }
})

// Update URL when filter sidebar is hidden
var filterOffcanvas = document.getElementById('offcanvasFilter');
filterOffcanvas.addEventListener('hide.bs.offcanvas', function () {
    updateURL();
})

// Update URL when search bar is clicked away from
function searchComplete() {
    var q = document.getElementsByClassName('search')[0].value;
    if (q.length > 0) {  // Include query in URL if active query
        searchParams.set('q', q);
    } else {
        searchParams.delete('q');
    }
    updateURL();
}

// Search, sort and filter by the current URL parameters
function applyURL() {
    // Get and apply sort
    var sort = searchParams.get('sort');
    if (sort) {
        document.getElementsByName('sort').forEach(
            function selectSort(elem) {
                if (elem.dataset['sort'] == sort) {
                    elem.checked = true;
                }
            }
        )
        resultsList.sort(sort, {order: searchParams.get('order')});
    }
    // Get and apply filters
    var filters = searchParams.getAll('f');
    if (filters.length > 0) {
        var cond = searchParams.get('c');
        if (cond === 'and') {
            document.getElementById('conditionand').checked = true;
        } else if (cond === 'or') {
            document.getElementById('conditionor').checked = true;
        }
        for (var i = 0, f; f = filters[i++];) {
            document.getElementById(f).checked = true;
        }
        applyFilters();
    }
    // Get and apply search
    var query = searchParams.get('q');
    if (query) {
        document.getElementsByClassName('search')[0].value = query;
        resultsList.search(query);
    }
}

// Update the URL with the current search parameters
function updateURL() {
    var query = searchParams.toString();
    if (query.length > 0) {  // Don't end the URL with '?'
        query = '?' + query;
    }
    if (window.location.search != query) {  // Update URL if changed
        history.replaceState(null, '', window.location.pathname + query);
    }
}

// Get the current sorting order from an active sort radio button
function getSortOrder(elem) {
    var fixedOrder = elem.dataset['order'];
    if (fixedOrder == 'asc' || fixedOrder == 'desc') {
        return fixedOrder;
    } else if (elem.classList.contains('desc')) {
        return 'desc';
    } else if (elem.classList.contains('asc')) {
        return 'asc';
    } else {
        return 'asc';
    }
}

function applyFilters() {
    searchParams.delete('f');
    searchParams.delete('c');
    var cond_and = document.getElementById('filterForm').elements['conditionand'].checked;
    var filters = [];
    var filterElements = document.getElementById('filterForm').getElementsByClassName('filter');
    for (var i = 0, elem; elem = filterElements[i++];) {
        if (elem.checked) {
            filters.push(elem.id);
            searchParams.append('f', elem.id);
        }
    }
    if (filters.length == 0) {
        resultsList.filter();  // Show all if nothing selected
        return countClasses();
    }
    searchParams.set('c', (cond_and) ? 'and' : 'or');
    resultsList.filter(function (item) {
        var inc = false;
        for (var i = 0, filt; filt = filters[i++];) {
            if (item.values()['filter-classes'].includes(filt)) {
                if (!cond_and) {
                    return true;
                }
                inc = true;
            } else {
                if (cond_and) {
                    return false;
                }
            }
        }
        return inc;
    });
    countClasses();
}

function resetFilters() {
    resultsList.filter();
    document.getElementById("filterForm").reset();
    countClasses();
    searchParams.delete('f');
    searchParams.delete('c');
}

function countClasses() {
    for (var i = 0, filt; filt = filterElements[i++];) {
        var count = 0;
        if (document.getElementById('filterForm').elements['conditionand'].checked) {
            var itms = resultsList.visibleItems;
        } else {
            var itms = resultsList.items;
        }
        for (var j = 0, itm; itm = itms[j++];) {
            if (itm.values()['filter-classes'].includes(filt.id)) {
                count++;
            }
        }
        var badge = filt.parentElement.getElementsByClassName('badge')[0];
        badge.innerHTML = count.toString();
    }
}

function warnIfNone() {
    if (resultsList.visibleItems.length === 0) {  // Show info box
        alertPlaceholder.innerHTML = '<div class="alert alert-info" role="alert">' +
            '<h4 class="alert-heading">No tests found</h4>' +
            '<p class="m-0">Try adjusting any active filters or searches, or ' +
            '<a href="javascript:clearAll()" class="alert-link">clear all</a>.</p>' +
            '</div>';
    } else {  // Remove info box
        alertPlaceholder.innerHTML = '';
    }
}

// Clear active search and filters
function clearAll() {
    document.getElementsByClassName('search')[0].value = '';
    resultsList.search('');
    searchParams.delete('q');
    resetFilters();
    updateURL();
}
